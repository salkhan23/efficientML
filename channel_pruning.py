# -------------------------------------------------------------------------------------------------
# Coarse Channel Pruning - Removing Channels
#
# Not only is the model smaller, it runs faster.
# But model structure is changed.
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from vgg import VGG
import train_cifar10


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

    model_acc = 81.06
    dense_model_acc = train_cifar10.evaluate(model, test_loader, device)
    print(f"Dense Model Accuracy {dense_model_acc:0.2f}")


if __name__ == "__main__":
    random_seed = 10
    saved_model_file = "./results_trained_models/07-06-2024_VGG_net_train_epochs_100_acc_81.pth"

    plt.ion()
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    print(f"GPU is available? {torch.cuda.is_available()}")

    net = VGG()
    main(net, saved_model_file)
