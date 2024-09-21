import matplotlib.pyplot as plt
import numpy as np

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


def get_model_sparsity(model):
    """
    Calculate the sparsity of a PyTorch model.

   - total_sparsity (float): The overall sparsity of the model.
    - sparsity_dict (dict): A dictionary mapping parameter names to their sparsity ratios.
    - n_param_dict (dict): A dictionary mapping parameter names to the number of parameters.
    """
    model_sparsity = 0
    total_param = 0

    sparsity_dict = {}
    n_param_dict = {}

    for name, param in model.named_parameters():
        if param.ndim > 1:
            n_param = param.numel()
            sparsity_param = get_tensor_sparsity(param)

            model_sparsity += sparsity_param * n_param
            total_param += n_param

            sparsity_dict[name] = sparsity_param
            n_param_dict[name] = n_param

    if total_param > 0:
        model_sparsity = model_sparsity / total_param

    return model_sparsity, sparsity_dict, n_param_dict


def plot_num_params_distribution(model):
    """
    Plot the distribution of the number of parameters in each layer of the model.

    :param model: The model whose parameters are to be plotted.

    :return: None
    """
    layer_names = []
    layer_num_params = []

    for name, param in model.named_parameters():
        if param.ndim > 1:
            layer_names.append(name)
            layer_num_params.append(param.numel())

    plt.figure(plt.figure(figsize=(10, 6)))
    plt.bar(np.arange(len(layer_names)), layer_num_params)
    plt.xlabel("Layer")
    plt.xticks(np.arange(len(layer_names)), layer_names, rotation=90)
    plt.ylabel("Number of parameters")
    plt.title("Number of parameters per layer")
