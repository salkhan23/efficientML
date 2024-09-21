# -------------------------------------------------------------------------------------------------
# Pruning Techniques and methods to find the best pruning ratio for individual layers of a model
#
# -------------------------------------------------------------------------------------------------
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import pickle
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import torch

import train_cifar10
from vgg import VGG
from model_analysis_utils import *


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

        # if name == 'backbone.conv1.weight':
        #     break

    return accuracies, sparsity_range, scanned_layer_names


def plot_sensitivity_scan_results_single_plot(sparsity_range, acc_per_layer, layer_names_list, full_model_acc):
    """

    :param sparsity_range:
    :param acc_per_layer:
    :param layer_names_list:
    :param full_model_acc:
    :return:
    """
    fig = plt.figure(figsize=(10, 6))

    for idx, acc_profile in enumerate(acc_per_layer):
        plt.plot(sparsity_range, acc_profile, label=layer_names_list[idx])

    plt.axhline(full_model_acc, label=f'full model acc {full_model_acc:0.2f}')

    plt.legend()
    plt.grid()

    plt.xlabel("Sparsity")
    plt.ylabel("Top 1 Accuracy")
    fig.suptitle("Sensitive Scan Analysis")

    plt.tight_layout()


def plot_sensitivity_scan_results_individual_layers(sparsity_range, acc_per_layer, layer_names_list, full_model_acc):
    """
    Plot pruning results for individual layers separately.

    :param sparsity_range: List or array of sparsity ratios
    :param acc_per_layer: List of lists or 2D array with accuracy values per layer
    :param layer_names_list: List of layer names
    :param full_model_acc: Accuracy of the full model

    :return: None
    """
    n_layers = len(acc_per_layer)
    subplot_dim = int(np.ceil(np.sqrt(n_layers)))

    fig, axes = plt.subplots(subplot_dim, subplot_dim, figsize=(15, 8))
    fig.suptitle("Sensitive Scan Analysis")

    axes = axes.ravel()
    for l_idx, layer_name in enumerate(layer_names_list):
        axes[l_idx].plot(sparsity_range, acc_per_layer[l_idx], color='blue')
        axes[l_idx].axhline(full_model_acc, label=f'full model acc {full_model_acc:0.2f}', color='r')
        axes[l_idx].set_xlabel("Sparsity ratio")
        axes[l_idx].set_ylabel("Top 1% accuracy")
        axes[l_idx].grid()
        axes[l_idx].set_title(layer_name)

    # Hide any unused subplots
    for ax in axes[n_layers:]:
        ax.set_visible(False)

    fig.tight_layout()


def find_layer_sparsity_ratios(
        target_model_sparsity, max_performance_drop, model, model_acc, sparsity_range, acc_per_layer,
        layer_name_list):
    """

    :param target_model_sparsity: n_zeros/ n_weights for all multi-dim model parameters
    :param max_performance_drop: # Max drop in performance allowed (percentage)
    :param model:
    :param model_acc: Total Dense (unpruned model) accuracy.
    :param sparsity_range: Sparsity Range the sensitivity for which acc_per_layer and layer_name_list is provided
    :param acc_per_layer:  List of accuracies for each model param in the sensitivity scan. Each
                           item in the list is another list of accuracies of that param for each
                           sparsity ratio in the sparsity range
    :param layer_name_list: List of model params in the  sensitivity scan result.
    :return:
    """
    acc_low_th = model_acc - max_performance_drop
    acc_per_layer = np.array(acc_per_layer)

    total_sparsity, sparsity_dict, n_params_dict = get_model_sparsity(model)

    def estimate_total_sparsity(sparsity_ratio_dict, num_params_dict):
        tot_sparsity = 0
        tot_n_params = 0
        for p_name, p_num_el in num_params_dict.items():
            tot_sparsity += p_num_el * sparsity_ratio_dict[p_name]
            tot_n_params += p_num_el

        if tot_n_params != 0:
            tot_sparsity = tot_sparsity / tot_n_params

        return tot_sparsity

    target_sparsities = copy.deepcopy(sparsity_dict)

    # Start pruning @ the deep-end, the least sensitive layers are usually there
    named_parameters = list(model.named_parameters())
    named_parameters = reversed(named_parameters)
    for name, param in named_parameters:
        if param.ndim > 1 and name in layer_name_list:

            results_idx = layer_name_list.index(name)  # index for the sparsity results

            layer_acc_above = np.where(acc_per_layer[results_idx, :] >= acc_low_th)[0]

            # if any sparsity ratio meets the criteria
            if layer_acc_above.size > 0:
                max_sparsity_idx = layer_acc_above[-1]  # find the Largest allowed sparsity ratio
                max_sparsity = sparsity_range[max_sparsity_idx]
                target_sparsities[name] = max_sparsity

                # check if the target sparsity for the whole model is met, exit if it is
                new_model_sparsity = estimate_total_sparsity(target_sparsities, n_params_dict)
                print(f"\tLayer'{name}' sparsity  = {target_sparsities[name]:0.2f}. "
                      f"[Total Model Sparsity {new_model_sparsity:0.2f}]")

                if new_model_sparsity >= target_model_sparsity:
                    print(f"Target Model Sparsity {target_model_sparsity:0.2f} Achieved")
                    break

    return target_sparsities


def fine_tune_model(model, device, train_loader, n_epochs, criterion, optimizer, lr_scheduler, pruning_cb):
    """
    Fine-tune a given model on a specified device using a training data loader.

    :param model:
    :param device:
    :param train_loader: DataLoader for the training dataset.
    :param n_epochs: Number of epochs to train the model.
    :param criterion: Loss function used for training.
    :param optimizer: Optimization algorithm used for training.
    :param lr_scheduler: Learning rate scheduler.
    :param pruning_cb: Callback function that applies a previously calculated pruning mask to maintain sparsity.

    :return:
    """
    model.to(device)
    # model.train()

    for epoch in range(n_epochs):

        epoch_loss = 0

        for b_idx, (data, labels) in tqdm(
                enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{n_epochs}', leave=False):

            data = data.to(device)
            labels = labels.to(device)

            model_out = model(data)

            loss = criterion(model_out, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()   # zero any previous gradients before backward pass
            loss.backward()         # Compute the gradients of the loss with respect to model parameters
            optimizer.step()        # Update the model parameters based on the computed gradients

            if pruning_cb is not None:
                pruning_cb(model)  # Apply the pruning mask to keep the model sparse after training,

        lr_scheduler.step()  # Drop learning rate according to lr scheduler, after each epoch

        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')


def main(model, dense_model_weights_file):
    """
    :param model: Trained model
    :param dense_model_weights_file:
    :return:
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    if not os.path.exists(dense_model_weights_file):
        raise FileNotFoundError(f"Cannot find stored model file {dense_model_weights_file}")

    net.load_state_dict(torch.load(dense_model_weights_file))
    model.to(device)

    results_store_dir = os.path.dirname(saved_model_file)

    b_size = 128
    data_dir = './data'

    # Data
    train_data_set, train_loader, test_data_set, test_loader, _ = (
        train_cifar10.get_cifar10_datasets_and_data_loaders(data_dir=data_dir, b_size=b_size))

    # model_acc = 81.06
    model_acc = train_cifar10.evaluate(model, test_loader, device)

    # Dense Model Details
    print(f"{'*' * 80}")
    print(f"Model: {model.__class__.__name__}")

    n_params = get_model_num_parameters(model)
    model_size = get_model_size(model)
    model_sparsity, _, _ = get_model_sparsity(model)
    print(
        f"Dense Model Details :\n"
        f"\tTop-1 Accuracy             : { model_acc :0.2f}\n"
        f"\tNumber of parameters       : {n_params}\n"
        f"\tSize                       : {model_size / MB:0.2f}MB\n"
        f"\tSparsity                   : {model_sparsity:0.2f}\n"
        f"\tWeights file size          : {os.path.getsize(dense_model_weights_file) / MB:0.3f} MB"
    )

    plot_model_weight_distribution(model)
    gcf = plt.gcf()
    gcf.suptitle("Dense model weight distribution")

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

    # Different sparsity ratio for each layer  -----------------------------------------------------
    print(f"{'*'*80}")
    sensitivity_scan_results_filename = 'sensitivity_results.pickle'
    sensitivity_scan_results_file = os.path.join(results_store_dir, sensitivity_scan_results_filename)

    if not os.path.exists(sensitivity_scan_results_file):
        print("Starting sensitivity scan ...")
        start_time = datetime.now()

        acc_per_layer, sparsity_range, layer_name_list = sensitivity_scan(model, device, test_loader)

        print(f"Sensitivity scan complete. Took {datetime.now() - start_time}")

        # Store the results for future results:
        with open(sensitivity_scan_results_file, 'wb') as f_handle:
            results_dict = {
                'acc_per_layer': acc_per_layer,
                'sparsity_range': sparsity_range,
                'layer_name_list': layer_name_list
            }
            pickle.dump(results_dict, f_handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Loading sensitivity scan results from {sensitivity_scan_results_file}")

        with open(sensitivity_scan_results_file, 'rb') as f_handle:
            results_dict = pickle.load(f_handle)
            acc_per_layer = results_dict['acc_per_layer']
            sparsity_range = results_dict['sparsity_range']
            layer_name_list = results_dict['layer_name_list']

    # Plot pruning results
    plot_sensitivity_scan_results_single_plot(sparsity_range, acc_per_layer, layer_name_list, model_acc)
    plot_sensitivity_scan_results_individual_layers(sparsity_range, acc_per_layer, layer_name_list, model_acc)

    # Plot the number of parameters per layer
    plot_num_params_distribution(model)

    # Given a min performance and a target sparsity ratio, find sparsity ratios of each layer
    # ---------------------------------------------------------------------------------------------
    target_model_sparsity = 0.90  # n_zeros/ n_weights
    max_performance_drop = 5  # Max drop in performance allowed (max accuracy = 100%)

    print(f"{'*' * 80}")
    print(f"Finding layer sparsity ratios to get a model with sparsity of {target_model_sparsity:0.2f}. "
          f"\nMax allowed performance Drop {max_performance_drop:0.2f}")

    sparse_dict = find_layer_sparsity_ratios(
        target_model_sparsity,
        max_performance_drop,
        model,
        model_acc,
        sparsity_range,
        acc_per_layer,
        layer_name_list
    )

    print(f"Pruning Model ...")
    pruner = FineGrainPruner(model, sparse_dict)
    with torch.no_grad():
        pruned_model_acc = train_cifar10.evaluate(model, test_loader, device)
    pruned_model_sparsity, _, _ = get_model_sparsity(model)

    print(
        f"Pruned Model Details (after pruning) :\n"
        f"\tTop-1 Accuracy             : {pruned_model_acc :0.2f}\n"
        f"\tSparsity                   : {pruned_model_sparsity:0.2f}")

    plot_model_weight_distribution(model)
    gcf = plt.gcf()
    gcf.suptitle(f"Weight distribution after finding individual param sparsities independently")

    # Fine Tune Pruned Model ----------------------------------------------------------------
    lr = 1e-3  # 1/100th of training lr
    n_epochs = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    print(f"{'*' * 80}")
    print(f"Fine Tuning Model [lr={lr}, n_epochs ={n_epochs}] ... ")

    fine_tune_model(
        model=model,
        device=device,
        train_loader=train_loader,
        n_epochs=n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        pruning_cb=pruner.apply_masks
    )

    with torch.no_grad():
        pruned_model_acc = train_cifar10.evaluate(model, test_loader, device)
    pruned_model_sparsity, _, _ = get_model_sparsity(model)

    # Space Savings -----------------------------------------------------------------------------
    # use pytorch's to_sparse method to save sparse weights compactly
    sparse_state_dict = {}
    for k, v in model.state_dict().items():
        if k in sparse_dict and sparse_dict[k] > 0:
            sparse_state_dict[k] = v.to_sparse()
        else:
            sparse_state_dict[k] = v

    sparse_weights_file = os.path.join(results_store_dir, 'sparse_weights.pth')
    torch.save(sparse_state_dict, sparse_weights_file)

    #  use pytorch's to_sparse method to save sparse weights compactly
    sparse_state_dict = {}
    for k, v in model.state_dict().items():
        if k in sparse_dict and sparse_dict[k] > 0:
            # Save the tensor as flattened 1D, convert to sparse
            v_flattened = v.flatten()
            v_sparse = v_flattened.to_sparse()
            sparse_state_dict[k] = {
                'shape': v.shape,
                'v_sparse': v_sparse
            }
        else:
            sparse_state_dict[k] = v

    sparse_weights_file = os.path.join(results_store_dir, 'sparse_weights.pth')
    torch.save(sparse_state_dict, sparse_weights_file)

    print(
        f"Pruned Model Details (after fine tuning) :\n"
        f"\tTop-1 Accuracy             : {pruned_model_acc :0.2f}\n"
        f"\tSparsity                   : {pruned_model_sparsity:0.2f}\n"
        f"\tSparse Weight file size    : {os.path.getsize(sparse_weights_file) / MB:0.3f} MB"
    )

    # load the saved model to see everything is working
    sparse_state_dict = torch.load(sparse_weights_file)
    reloaded_dense_state_dict = {}

    for k, v in sparse_state_dict.items():
        if isinstance(v, dict) and 'v_sparse' in v:
            # If the weight is sparse
            sparse_tensor = v['v_sparse'].to_dense()
            shape = tuple(v['shape'])
            reloaded_dense_state_dict[k] = torch.reshape(sparse_tensor, shape)
        else:
            # If the weight is dense
            reloaded_dense_state_dict[k] = v

    net2 = VGG()
    net2.load_state_dict(reloaded_dense_state_dict)
    net2.to(device)

    with torch.no_grad():
        reloaded_model_acc = train_cifar10.evaluate(net2, test_loader, device)
    reloaded_model_sparsity, _, _ = get_model_sparsity(net2)

    print(
        f"Reloaded Sparse Model :\n"
        f"\tTop-1 Accuracy             : {reloaded_model_acc :0.2f}\n"
        f"\tSparsity                   : {reloaded_model_sparsity:0.2f}\n"
    )

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    random_seed = 10
    saved_model_file = "./results_trained_models/07-06-2024_VGG_net_train_epochs_100_acc_81.pth"

    plt.ion()
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    print(f"GPU is available? {torch.cuda.is_available()}")

    net = VGG()
    main(net, saved_model_file)
