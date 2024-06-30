# -------------------------------------------------------------------------------------------------
# Pruning Techniques and methods to find the best pruning ratio for individual layers of a model
#
# -------------------------------------------------------------------------------------------------
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torchvision

import train_cifar10
from vgg import VGG

Byte = 8
KB = Byte * 1024
MB = KB * 1024
GB = MB * 1024


def get_model_num_parameters(model):
    """
    Count the number of parameters in a model
    :param model:
    :return:
    """
    n_params = 0
    for param in model.parameters():
        p_shape = param.size()
        p_n_elements = 1
        for dim in p_shape:
            p_n_elements *= dim

        n_params += p_n_elements

    return n_params


def get_model_size(model, data_width=32):
    n_params = get_model_num_parameters(model) * data_width
    return n_params


def get_tensor_sparsity(tensor):
    """
    Sparsity defined as n_zeros/n_elements
    :param tensor:
    :return:
    """
    return 1 - (tensor.count_nonzero() / tensor.numel())


def plot_model_weight_distribution(model, bins=256):
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()

    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True, color='blue', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1

    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor

    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
                     = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)

    # Extreme Cases
    if sparsity == 1.0:
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    n_elements = tensor.numel()  # number of elements
    n_zeros = n_elements - tensor.count_nonzero()
    # print(f"pre-pruning sparsity {get_tensor_sparsity(tensor)}")

    # Magnitude based pruning
    importance = torch.abs(tensor)

    # Calculate the pruning threshold - all synapses w/ importance < threshold will be removed
    n_zeros_want = round(n_elements * sparsity)  # Number of non-zeros to get the desired sparsity
    n_non_zeros_want = n_elements - n_zeros_want
    if n_zeros > n_zeros_want:
        return torch.ones_like(tensor)  # already below target sparsity

    # Get the indices of weights to keep
    imp_flattened = importance.flatten()
    values, indexes = torch.topk(imp_flattened, n_non_zeros_want)

    # Create a mask
    th = min(values)
    # mask = torch.gt(abs(tensor), th * torch.ones_like(tensor))
    # torch.gt only does >. Find all places th > abs(tensor), then reverse it
    mask = torch.gt(th * torch.ones_like(tensor), abs(tensor))
    mask = ~mask

    # inplace multiple with the mask . Detach needed for in-place operation.
    with torch.no_grad():
        tensor.detach()
        tensor.mul_(mask)

    # print(f" after pruning sparsity {get_tensor_sparsity(tensor)}")

    return mask


class FineGrainPruner:
    def __init__(self, model, sparsity_dict):
        """
        Apply fine grain (element-wise pruning) on each named model parameter in the sparsity dictionary

        :param model:
        :param sparsity_dict: a dictionary with keys model names and value equal to the pruning ratio
        """
        self.masks = self.prune(model, sparsity_dict)

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        """
        Prune the Model

        :param model:
        :param sparsity_dict:
        :return:
        """
        masks = {}

        for name, param in model.named_parameters():
            print(name)
            sparsity_ratio = sparsity_dict.get(name, None)
            if sparsity_ratio is not None:
                masks[name] = fine_grained_prune(param, sparsity_ratio)

        return masks

    @torch.no_grad()
    def apply_masks(self, model):
        """ Apply pruning masks to model """
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]


def sensitivity_scan(model, device, test_loader, scan_step=0.1, scan_start=0.4, scan_end=1.0):
    """
    Perform sensitivity analysis by scanning through different sparsity levels for each layer's weights.

    :param model: The neural network model.
    :param device: The device to run the model on (e.g., 'cpu', 'cuda').
    :param test_loader: The data loader for the test dataset.
    :param scan_step: The step size for scanning sparsity levels, defaults to 0.1.
    :param scan_start: The starting sparsity level, defaults to 0.4.
    :param scan_end: The ending sparsity level (exclusive), defaults to 1.0.
    :return: Tuple containing accuracies per layer, sparsity range, and names
    """
    sparsity_range = np.arange(scan_start, scan_end, scan_step)

    accuracies = []  # per layer, each column is different sparsity ratio
    scanned_layer_names = []

    for name, param in model.named_parameters():

        if param.ndim > 1:
            print(f"Analyzing Layer {name}.")
            print(f"\tstart sparsity {get_tensor_sparsity(param)}. ")

            org_param = param.detach().clone()
            layer_accuracies = []

            for sparsity in sparsity_range:
                fine_grained_prune(param, sparsity)

                with torch.no_grad():
                    acc = train_cifar10.evaluate(model, test_loader, device)

                layer_accuracies.append(acc)
                param.data.copy_(org_param.data)  # restore the original weights

                # print(f"\tstart sparsity {get_tensor_sparsity(param)}. ")
                # print(f"\t    @ sparsity {get_tensor_sparsity(param)}. Accuracy {acc:.2f}")

            accuracies.append(layer_accuracies)
            scanned_layer_names.append(name)

            print(f"\tLayer '{name}' accuracies: {', '.join([f'{acc:.2f}' for acc in layer_accuracies])}")
            print(f"\tFinal sparsity: {get_tensor_sparsity(param):.2f}")

    return accuracies, sparsity_range, scanned_layer_names


def main(model):
    """

    :param model: Trained model
    :return:
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    b_size = 128
    data_dir = './data'

    # Data
    train_data_set, train_loader, test_data_set, test_loader, _ = (
        train_cifar10.get_cifar10_datasets_and_data_loaders(data_dir=data_dir, b_size=b_size))

    acc = 81.06
    # acc = train_cifar10.evaluate(model, test_loader, device)

    # Dense Model Details
    print(f"Model: {model.__class__.__name__}")

    n_params = get_model_num_parameters(model)
    model_size = get_model_size(model)
    print(
        f"Dense Model Acc {acc:0.2f}, Model Size {model_size / MB:0.2f}MB. "
        f"Number of parameters {n_params} ")
    plot_model_weight_distribution(model)
    gcf = plt.gcf()
    gcf.suptitle("Dense Model Weight Distribution")

    # # Prune a random layer # ------------------------------------------------------------------------
    # layer_to_prune = "backbone.conv1"
    # sparsity = 0.75
    #
    # model_layers = dict(model.named_modules())  # creates a dictionary of model layers
    #
    # if layer_to_prune in model_layers:
    #     layer = model_layers[layer_to_prune]
    #     layer_w = layer.weight
    #
    #     print(f"Prune Layer '{layer_to_prune}'. Target Sparsity: {sparsity}.")
    #
    #     start_sparsity = get_tensor_sparsity(layer_w)
    #     fine_grained_prune(layer_w, sparsity)
    #     end_sparsity = get_tensor_sparsity(layer_w)
    #     print(f"Done. Start Sparsity {start_sparsity}. Final Sparsity {end_sparsity}")
    # else:
    #     raise ValueError(f"Layer {layer_to_prune} not in model")

    # # Prune the whole model with a single sparsity ratio ------------------------------------------
    # sparsity = 0.75
    #
    # sparse_dict = {name: sparsity for name, param in model.named_parameters()}
    # FineGrainPruner(model, sparse_dict)
    #
    # plot_model_weight_distribution(model)
    # gcf = plt.gcf()
    # gcf.suptitle(f"Weight distribution After pruning with fix sparsity {sparsity}")

    # Different running ratio for each layer  -----------------------------------------------------
    acc_per_layer, sparsity_range, layer_name_list = sensitivity_scan(model, device, test_loader)

    plt.figure(figsize=(9, 9))
    for idx, acc_profile in enumerate(acc_per_layer):
        plt.plot(sparsity_range, acc_profile, label=layer_name_list[idx])
    plt.legend()

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    random_seed = 10
    saved_model_file = "./results_trained_models/07-06-2024_VGG_net_train_epochs_100_acc_81.pth"

    plt.ion()
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    print(f"GPU is available? {torch.cuda.is_available()}")

    # load the model
    if not os.path.exists(saved_model_file):
        raise FileNotFoundError(f"Cannot find stored model file {saved_model_file}")

    net = VGG()
    net.load_state_dict(torch.load(saved_model_file))

    main(net)
