# -------------------------------------------------------------------------------------------------
# Coarse Channel Pruning - Removing Channels
#
# Not only is the model smaller, it runs faster.
# But model structure is changed.
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchprofile import profile_macs

from vgg import VGG
import train_cifar10
import model_analysis_utils
import pruning


@torch.no_grad()
def sort_channels_on_importance(model):
    """
    Sort channels of each convolutional (and subsequent BN) layers based on importance criterion.
    Makes Pruning easier to implement.

    Assumes the channel has a backbone part. Only the backbone component of the model is changed
    Assumes that each convolutional layer is followed by a BN layer.
    """

    modified_model = copy.deepcopy(model)  # do not modify the original model, deep copy it.

    all_convs = [m for m in modified_model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in modified_model.backbone if isinstance(m, nn.BatchNorm2d)]

    def get_input_channel_importance(weight):
        """ Find the Euclidian norm of each input channel of a convolutional layer """

        squared_weight = weight ** 2
        squared_sum = squared_weight.sum(dim=(0, 2, 3))  # dim 1 = input channels
        imp = torch.sqrt(squared_sum)

        return imp

    for i_conv in range(len(all_convs) - 1):
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the previous BN layer
        # - the input dimension of the next conv (we compute importance here)
        curr_conv = all_convs[i_conv]
        curr_bn = all_bns[i_conv]
        next_conv = all_convs[i_conv + 1]

        importance = get_input_channel_importance(next_conv.weight)

        # sort importance according to descending order
        values, indexes = torch.sort(importance, descending=True)

        # Arrange the output channels of the current channel according to indexes
        curr_conv.weight.copy_(torch.index_select(curr_conv.weight.detach(), 0, indexes))

        # Adjust the current Batch Normalization layer
        for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
            tensor_to_apply = getattr(curr_bn, tensor_name)
            tensor_to_apply.copy_(
                torch.index_select(tensor_to_apply.detach(), 0, indexes)
            )

        # adjust the inputs to the next layer
        next_conv.weight.copy_(
            torch.index_select(next_conv.weight.detach(), 1, indexes)
        )

    return modified_model


@torch.no_grad()
def prune_model_channels(model, sparsity):
    """
     Prunes the channels of the convolutional and batch normalization layers of a model.

    :param model: The model to prune.
    :param sparsity: The fraction of channels to prune (between 0 and 1).
    :return: The pruned model.
    """
    sparsity = min(max(0.0, sparsity), 1.0)

    modified_model = copy.deepcopy(model)  # prevent overwrite

    all_convs = [m for m in modified_model.backbone if isinstance(m, nn.Conv2d)]
    all_bns = [m for m in modified_model.backbone if isinstance(m, nn.BatchNorm2d)]

    for c_layer_idx in range(len(all_convs) - 1):
        curr_conv = all_convs[c_layer_idx]
        curr_bn = all_bns[c_layer_idx]
        next_conv = all_convs[c_layer_idx + 1]

        # Figure out the number of output channels to keep
        n_ch_out, n_ch_in, n_r, n_c = curr_conv.weight.shape
        n_ch_out_keep = int(np.round(n_ch_out * (1 - sparsity)))

        curr_conv.weight.set_(curr_conv.weight.detach()[:n_ch_out_keep, ])  # output channels are removed

        # Change the Batch Normalization layers as well
        curr_bn.weight.set_(curr_bn.weight.detach()[:n_ch_out_keep, ])
        curr_bn.bias.set_(curr_bn.bias.detach()[:n_ch_out_keep, ])
        curr_bn.running_mean.set_(curr_bn.running_mean.detach()[:n_ch_out_keep,])
        curr_bn.running_var.set_(curr_bn.running_var.detach()[:n_ch_out_keep, ])

        # Change the input dimensions of the next convolutional layer
        next_conv.weight.set_(next_conv.weight.detach()[:, :n_ch_out_keep, ])

    return modified_model


def main(model, dense_model_weights_file):
    """

    :param model:
    :param dense_model_weights_file:
    :return:
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(dense_model_weights_file):
        raise FileNotFoundError(f"Cannot find stored model file {dense_model_weights_file}")

    net.load_state_dict(torch.load(dense_model_weights_file))
    model.to(device)

    b_size = 128
    data_dir = './data'

    # Data
    train_data_set, train_loader, test_data_set, test_loader, _ = (
        train_cifar10.get_cifar10_datasets_and_data_loaders(data_dir=data_dir, b_size=b_size))

    # model_acc = 81.06
    dense_model_acc = train_cifar10.evaluate(model, test_loader, device)
    print(f"Dense Model Accuracy {dense_model_acc:0.2f}")

    # Channel Pruning ------------------------------------------------------------------------
    # Sort channels according to importance, to make it easier to prune
    sorted_model = sort_channels_on_importance(model)

    sorted_model_acc = train_cifar10.evaluate(model, test_loader, device)
    print(f"Sorted Model Accuracy {sorted_model_acc:0.2f}")

    # Prune the Channels according to  a sparsity ratio
    sparsity = 0.3
    pruned_model = prune_model_channels(sorted_model, sparsity)

    pruned_model_acc = train_cifar10.evaluate(pruned_model, test_loader, device)
    print(f"Pruned Model Accuracy {pruned_model_acc:0.2f}")

    # Fine Tune Pruned Model ----------------------------------------------------------------
    lr = 1e-3  # 1/100th of training lr
    n_epochs = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    print(f"{'*' * 80}")
    print(f"Fine Tuning Model [lr={lr}, n_epochs ={n_epochs}] ... ")

    pruning.fine_tune_model(
        model=model,
        device=device,
        train_loader=train_loader,
        n_epochs=n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        pruning_cb=None
    )

    pruned_model_acc = train_cifar10.evaluate(model, test_loader, device)
    print(f"Fine Tune Model Accuracy {pruned_model_acc:0.2f}")

    # Show improvements due to Pruning  ---------------------------------------------------
    # Model Size Reduction
    dense_model_size = model_analysis_utils.get_model_size(model)
    pruned_model_size = model_analysis_utils.get_model_size(pruned_model)
    print(
        f"Dense model size {dense_model_size/model_analysis_utils.MB:0.2f}. Pruned Model Size "
        f"{pruned_model_size/model_analysis_utils.MB:0.2f}. "
        f"Reduction {pruned_model_size/dense_model_size * 100:0.2f}%")

    image_idx = int(torch.rand(1) * len(test_data_set.data))
    test_image = test_data_set.data[image_idx]
    test_image = torch.from_numpy(test_image).float()  # Model expects a float time
    test_image = torch.permute(test_image, (2, 0, 1))  # channel first
    test_image = test_image.unsqueeze(dim=0)  # Add batch dimension
    test_image = test_image.to(device)

    print(f"Testing using image @ index {image_idx}")

    # Model MAC reduction Multiply and accumulate operations
    dense_model_mac = profile_macs(model, test_image)
    prune_model_mac = profile_macs(pruned_model, test_image)

    print(
        f"Dense  model MACs {dense_model_mac / 1e6:0.2f}, "
        f"Pruned Model MACs {prune_model_mac / 1e6:0.2f}. "
        f"Reduction {prune_model_mac / dense_model_mac * 100:0.2f} %")

    # Timing Improvement
    @torch.no_grad()
    def measure_latency(model1, input1, n_warmup=20, n_test=100):
        model1.eval()
        # warmup  - Eliminate any setup latency issue
        for _ in range(n_warmup):
            _ = model1(input1)
        # real test
        t1 = time()
        for _ in range(n_test):
            _ = model1(input1)
        t2 = time()

        return (t2 - t1) / n_test  # average latency

    dense_model_latency = measure_latency(model, test_image)
    prune_model_latency = measure_latency(pruned_model, test_image)

    print(
        f"Dense model latency {dense_model_latency*1000:0.2f} ms. "
        f"Pruned Model MACs {prune_model_latency*1000:0.2f} ms. "
        f"Reduction {(dense_model_latency - prune_model_latency) / dense_model_latency * 100:0.2f} %")

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
