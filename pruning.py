import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torchvision

import train
from vgg import VGG


def evaluate_cifar_10(model, device, b_size=128, num_workers=12):
    data_dir = './data/cifar10'

    transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=b_size,
        shuffle=False,
        num_workers=num_workers)

    model = model.to(device)
    acc = train.evaluate(model, test_loader, device)
    print(f"Cifar-10 test accuracy {acc:0.2f}")


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


def main(model):
    """

    :param model: Trained model
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # evaluate_cifar_10(model, device)

    n_params = get_model_num_parameters(model)
    print(f" Model {model.__class__.__name__} has {n_params} parameters")

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    random_seed = 10
    saved_model_file = "./results_trained_models/07-06-2024_VGG_net_train_epochs_100_acc_81.pth"

    plt.ion()
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    if torch.cuda.is_available():
        print("CUDA is available. Moving model to GPU...")
    else:
        print("CUDA is not available. Moving model to CPU...")

    # ---------------------------------------------------------------
    # load the model
    if not os.path.exists(saved_model_file):
        raise FileNotFoundError(f"Cannot find stored model file {saved_model_file}")

    net = VGG()
    net.load_state_dict(torch.load(saved_model_file))

    main(net)
