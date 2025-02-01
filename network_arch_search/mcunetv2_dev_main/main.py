import argparse
import json
from PIL import Image
from tqdm import tqdm
import copy
import math
import numpy as np
import os
import random
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import datasets, transforms

from mcunet.tinynas.search.accuracy_predictor import AccuracyDataset, MCUNetArchEncoder
from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
from mcunet.utils.mcunet_eval_helper import calib_bn, validate
from mcunet.utils.arch_visualization_helper import draw_arch
from mcunet.utils.pytorch_utils import count_peak_activation_size, count_net_flops, count_parameters


def build_val_data_loader(data_dir, resolution, batch_size=128, split=0):
    """
    We use split = 0 (default value) to represent the validation set (cannot be directly used for architecture search),
    and split = 1 will be used as a holdout minival set (used to generate the accuracy dataset and calibrate
    BN parameters).

    :param data_dir:
    :param resolution:
    :param batch_size:
    :param split: split = 0: real val set, split = 1: holdout validation set
    :return:
    """
    assert split in [0, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    kwargs = {
        "num_workers": min(8, os.cpu_count()),
        "pin_memory": False
    }

    val_transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize,
        ]
    )

    # The ImageFolder dataset expects the data_dir directory to be organized in a specific way:
    #
    # data_dir/
    #     class_1/
    #         img1.jpg
    #         img2.jpg
    #         ...
    #     class_2/
    #         img1.jpg
    #         img2.jpg
    #         ...
    #
    # The subdirectory names (class_1, class_2, etc.) are treated as the class labels.
    # Each image is associated with a label corresponding to its subdirectory.
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    # This selects a subset of indices from the dataset for the validation split.
    # Even or Odd image sets.
    val_dataset = torch.utils.data.Subset(
        val_dataset, list(range(len(val_dataset)))[split::2]  # python slicing sequence [start:stop:step]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    return val_loader


def evaluate_sub_network(ofa_network1, cfg1, data_dir, image_size1=None):
    """
    Evaluates a subnetwork sampled from a larger once-for-all (OFA) network.

    This function performs the following steps:
    1. Samples the active subnet from the OFA network based on the provided configuration.
    2. Extracts the subnet with corresponding weights from the OFA network.
    3. Computes efficiency metrics for the subnet, including peak memory usage, multiply-accumulate operations (MACs),
       and the number of parameters.
    4. Recalibrates the Batch Normalization (BN) parameters of the subnet to align with its activation patterns
       using a calibration dataset.
    5. Defines a validation data loader based on the provided dataset directory and image size.
    6. Evaluates the accuracy of the subnet using the validation dataset.

    Args:
        ofa_network1 (object): The once-for-all (OFA) network from which the subnet is sampled.
        cfg1 (dict): Configuration for sampling the active subnet. Can include parameters like image size.
        data_dir (str): Directory containing the dataset used for BN calibration and validation.
        image_size1 (int, optional): The input image size. If not provided, it will be extracted from `cfg1`.

    Returns:
        tuple: A tuple containing the following metrics:
            - acc1 (float): The accuracy of the subnet on the validation dataset.
            - peak_memory (int): The peak activation memory usage of the subnet.
            - macs (int): The number of multiply-accumulate operations (MACs) required by the subnet.
            - params1 (int): The number of parameters in the subnet.
    """

    if "image_size" in cfg1:
        image_size1 = cfg1["image_size"]

    batch_size = 128

    # step 1. sample the active subnet with the given config.
    ofa_network1.set_active_subnet(**cfg1)

    # step 2. extract the subnet with corresponding weights.
    # Returns a new network with the dimensions (and weights) of the active subnet of the ofa_network
    subnet = ofa_network1.get_active_subnet().to(device)

    # step 3. calculate the efficiency stats of the subnet.
    peak_memory1 = count_peak_activation_size(subnet, (1, 3, image_size1, image_size1))

    macs1 = count_net_flops(subnet, (1, 3, image_size1, image_size1))

    params1 = count_parameters(subnet)

    # step 4. perform BN parameter re-calibration.
    # BN calibration is necessary when evaluating subnets of a larger model because the BN layers in the full model
    # store statistics (mean and variance) based on the full architecture, which may not align with the activation
    # distributions of the smaller subnet. Without recalibration, the subnet may misnormalize activations,
    # leading to incorrect outputs and degraded performance.
    calib_bn(subnet, data_dir, batch_size, image_size1)

    # step 5. define the validation dataloader.
    val_loader = build_val_data_loader(data_dir, image_size1, batch_size)

    # step 6. validate the accuracy.
    acc1 = validate(subnet, val_loader)

    return acc1, peak_memory1, macs1, params1


def visualize_subnet(cfg1):
    draw_arch(cfg1["ks"], cfg1["e"], cfg1["d"], cfg1["image_size"], out_name="viz/subnet")
    im = Image.open("viz/subnet.png")
    im = im.rotate(90, expand=1)
    plt.figure(figsize=(im.size[0] / 250, im.size[1] / 250))
    plt.axis("off")
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    plt.ion()
    random_seed = 10
    torch.random.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_directory = "/home/salman/workspace/pytorch/efficientML/data/visual_wake_words/vww-s256/val"

    # -----------------------------------------------------------------------------------
    # Plot sample images from visual wake words dir
    print("Plotting sample images from the visual wake woods dataset ...")

    val_data_loader = build_val_data_loader(data_directory, resolution=128, batch_size=1)

    vis_x, vis_y = 2, 3
    fig, axs = plt.subplots(vis_x, vis_y)

    num_images = 0
    for data, label in val_data_loader:
        img = np.array((((data + 1) / 2) * 255).numpy(), dtype=np.uint8)
        img = img[0].transpose(1, 2, 0)

        if label.item() == 0:
            label_text = "No person"
        else:
            label_text = "Person"

        axs[num_images // vis_y][num_images % vis_y].imshow(img)
        axs[num_images // vis_y][num_images % vis_y].set_title(f"Label: {label_text}")
        axs[num_images // vis_y][num_images % vis_y].set_xticks([])
        axs[num_images // vis_y][num_images % vis_y].set_yticks([])
        num_images += 1
        if num_images > vis_x * vis_y - 1:
            break

    # -----------------------------------------------------------------------------------
    # Create a Once-for-all network
    # -----------------------------------------------------------------------------------
    print("Creating sample OFA MCU Networks ...")

    ofa_network = OFAMCUNets(
        n_classes=2,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.0,
        base_stage_width="mcunet384",
        width_mult_list=[0.5, 0.75, 1.0],
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[0, 1, 2],
        base_depth=[1, 2, 2, 2, 2],
        fuse_blk1=True,
        se_stages=[False, [False, True, True, True], True, True, True, False],
    )

    ofa_network.load_state_dict(
        torch.load("vww_supernet.pth", map_location="cpu", weights_only=True)["state_dict"], strict=True,
    )

    ofa_network = ofa_network.to(device)

    # ---------------------------------------------------------------------------------
    # Sample Some Networks and visualize them
    # ---------------------------------------------------------------------------------
    image_size = 256

    sample_function = random.choice
    cfg = ofa_network.sample_active_subnet(sample_function, image_size=image_size)
    acc, _, _, params = evaluate_sub_network(ofa_network, cfg, data_dir=data_directory)
    visualize_subnet(cfg)
    print(f"The accuracy of the {sample_function.__name__} sampled subnet: "
          f"#params={params / 1e6: .1f}M, accuracy={acc: .1f}%.")

    sample_function = max
    largest_cfg = ofa_network.sample_active_subnet(sample_function, image_size=image_size)
    acc, _, _, params = evaluate_sub_network(ofa_network, largest_cfg, data_dir=data_directory)
    visualize_subnet(largest_cfg)
    print(f"The {sample_function.__name__} subnet: #params={params / 1e6: .1f}M, accuracy={acc: .1f}%.")

    sample_function = min
    smallest_cfg = ofa_network.sample_active_subnet(sample_function, image_size=image_size)
    acc, peak_memory, macs, params = evaluate_sub_network(ofa_network, smallest_cfg, data_dir=data_directory)
    visualize_subnet(smallest_cfg)
    print(f"The {sample_function.__name__} subnet: #params={params / 1e6: .1f}M, accuracy={acc: .1f}%.")
    print(f"peak memory {peak_memory/1024:0.2f}KB, macs {macs/10e6:0.2f}M")

    import pdb
    pdb.set_trace()
