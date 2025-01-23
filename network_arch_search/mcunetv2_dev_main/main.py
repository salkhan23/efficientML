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
    # Even
    val_dataset = torch.utils.data.Subset(
        val_dataset, list(range(len(val_dataset)))[split::2]  # python slicing sequence [start:stop:step]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    return val_loader


if __name__ == "__main__":
    plt.ion()
    random_seed = 10
    torch.random.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_directory = "/home/salman/workspace/pytorch/efficientML/data/visual_wake_words/vww-s256/val"

    # -----------------------------------------------------------------------------------
    # Plot sample images from visual wake words dir
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
        torch.load("vww_supernet.pth", map_location="cpu")["state_dict"], strict=True
    )

    ofa_network = ofa_network.to(device)

    import pdb
    pdb.set_trace()
