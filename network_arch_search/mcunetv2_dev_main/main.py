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


class AnalyticalEfficiencyPredictor:
    """
     Evaluates the efficiency of a given model by estimating:
        - Peak Memory Utilization (in KB)
        - Number of Multiply-Accumulate Operations (MACs)
    """
    def __init__(self, net):
        self.net = net

    def get_efficiency(self, spec: dict):
        """
        Computes efficiency metrics (MACs and Peak Activation Memory) for a given network specification.

        :param spec: A dictionary containing model specifications, including 'image_size'
        :return:A dictionary  containing Efficiency metrics including:
            'millionMACs' : Total MACs divided by 1e6 (millions of MACs).
            'KBPeakMemory': Peak memory usage in KB.
        """
        self.net.set_active_subnet(**spec)  # configure the OFA MCU Net with the sample network
        subnet = self.net.get_active_subnet()  # create a new subnet from the active subnet of the OFA.

        if torch.cuda.is_available():
            subnet = subnet.cuda()

        image_size1 = spec['image_size']
        data_shape = (1, 3, image_size1, image_size1)
        macs1 = count_net_flops(subnet, data_shape)

        peak_memory1 = count_peak_activation_size(subnet, (1, 3, image_size1, image_size1))

        return dict(millionMACs=macs1 / 1e6, KBPeakMemory=peak_memory1 / 1024)

    @staticmethod
    def satisfy_constraint(measured: dict, target: dict):
        for key in measured:
            # if the constraint is not specified, we just continue
            if key not in target:
                continue
            # if we exceed the constraint, just return false.
            if measured[key] > target[key]:
                return False
        # no constraint violated, return true.
        return True


class AccuracyPredictor(nn.Module):
    """
    A neural network module to predict the accuracy of OFA MCU net subnetworks.

    The predictor takes a binary feature vector (obtained by encoding an architecture via an
    MCUNetArchEncoder) and estimates its accuracy. The feature vector is processed through a
    multilayer perceptron (MLP) with ReLU activations

    The architecture for the predictor consists of:
      - A configurable number of fully connected layers (n_layers) with hidden_size neurons.
      - ReLU activation functions between layers.
      - A final linear layer that outputs a single scalar value (the predicted accuracy).

    """
    def __init__(
        self,
        arch_encoder_net,
        hidden_size=400,
        n_layers=3,
        checkpoint_path=None,
        device1="cuda:0",
    ):
        """
        Initializes the AccuracyPredictor.

        :param arch_encoder_net: An instance of MCUNetArchEncoder used to encode architectures into feature vectors.
        :param hidden_size: The number of neurons in each hidden layer of the MLP. Default is 400.
        :param n_layers: The number of linear layers in the MLP. Default is 3. Note that the first layer's input\n
            dimension is determined by arch_encoder_net.n_dim.
        :param checkpoint_path: Path to a pretrained checkpoint to load weights from (if available).\n
        :param device1: The device to use for computations (e.g., 'cuda:0' or 'cpu').\n
        """
        super(AccuracyPredictor, self).__init__()

        self.arch_encoder = arch_encoder_net
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device1

        layers = []

        # Each layer (nn.Linear) has hidden_size channels & uses nn.ReLU as the activation function.
        # Use self.arch_encoder.n_dim to get the input dimension
        for i in range(self.n_layers):
            if i == 0:
                layers.append(nn.Linear(arch_encoder_net.n_dim, self.hidden_size))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_size, 1, bias=False))

        self.layers = nn.Sequential(*layers)

        self.base_acc = nn.Parameter(
            torch.zeros(1, device=self.device), requires_grad=False
        )

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.load_state_dict(checkpoint)
            print("Loaded checkpoint from %s" % checkpoint_path)

        self.layers = self.layers.to(self.device)

    def forward(self, x):
        y = self.layers(x).squeeze()
        return y + self.base_acc

    def predict_acc(self, arch_dict_list):
        # get feature vectors for each arch dict in arch_dict_list
        x = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]

        # convert to tensor. THis creates a batch input. [num arch_dict, feature vectors]. THe way forward likes it.
        x = torch.tensor(np.array(x)).float().to(self.device)

        return self.forward(x)


class RandomSearcher:
    def __init__(self, efficiency_predictor1, accuracy_predictor):

        self.efficiency_predictor = efficiency_predictor1
        self.accuracy_predictor = accuracy_predictor

    def random_valid_sample(self, constraint):
        # randomly sample subnets until finding one that satisfies the constraint
        while True:
            # Dictionary of network architecture, sampled from max configs specified in arch_encoder
            sample = self.accuracy_predictor.arch_encoder.random_sample_arch()

            efficiency = self.efficiency_predictor.get_efficiency(sample)

            if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
                return sample, efficiency

    def run_search(self, constraint, n_subnets=100):
        subnet_pool = []

        # sample subnets
        for _ in tqdm(range(n_subnets)):
            sample, efficiency = self.random_valid_sample(constraint)  # Returns dictionaries of architectures
            subnet_pool.append(sample)

        # predict the accuracy of subnets
        accs = self.accuracy_predictor.predict_acc(subnet_pool)
        # get the index of the best subnet
        best_idx = torch.argmax(accs)

        # return the best subnet
        return accs[best_idx], subnet_pool[best_idx]


def search_and_measure_acc(agent, constraint, data_dir, **kwargs):

    n_subnets = kwargs.get("n_subnets", None)
    if n_subnets is not None:
        best_info = agent.run_search(constraint, n_subnets)
    else:
        best_info = agent.run_search(constraint)

    # get searched subnet
    ofa_network.set_active_subnet(**best_info[1])
    subnet = ofa_network.get_active_subnet().to(device)

    # calibrate bn
    calib_bn(subnet, data_dir, best_info[1]["image_size"], 128)

    # build val loader
    val_loader = build_val_data_loader(data_dir, best_info[1]["image_size"], 128)

    # measure accuracy
    acc1 = validate(subnet, val_loader)

    # print best_info
    print(f"Accuracy of the selected subnet: {acc1}")

    # visualize model architecture
    visualize_subnet(best_info[1])

    return acc1, subnet


if __name__ == "__main__":
    plt.ion()
    random_seed = 10
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)

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
    # Sample some networks and visualize them
    # ---------------------------------------------------------------------------------
    image_size = 256
    print(f"Getting accuracy's of sampled networks. Image Resolution {image_size}x{image_size}")

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
    print(f"peak memory {peak_memory/1024:0.2f}KB, macs {macs/1e6:0.2f}M")

    # Analytical Calculation of model efficiency (Peak activation memory & MACs)
    # ---------------------------------------------------------------
    efficiency_predictor = AnalyticalEfficiencyPredictor(ofa_network)

    smallest_cfg = ofa_network.sample_active_subnet(sample_function=min, image_size=image_size)
    eff_smallest = efficiency_predictor.get_efficiency(smallest_cfg)
    print(f"Efficiency stats of the smallest subnet: {eff_smallest}")

    largest_cfg = ofa_network.sample_active_subnet(sample_function=max, image_size=image_size)
    eff_largest = efficiency_predictor.get_efficiency(largest_cfg)
    print(f"Efficiency stats of the largest subnet: {eff_largest}")

    # ---------------------------------------------------------------------------------
    # Accuracy Prediction Network
    # ---------------------------------------------------------------------------------
    image_size_list = [96, 112, 128, 144, 160]

    # A Class that can OFA MCU Net architectures to compact feature vectors (required by the accuracy dataset)
    arch_encoder = MCUNetArchEncoder(
        image_size_list=image_size_list,
        base_depth=ofa_network.base_depth,
        depth_list=ofa_network.depth_list,
        expand_list=ofa_network.expand_ratio_list,
        width_mult_list=ofa_network.width_mult_list,
    )

    # Create accuracy predictor class
    os.makedirs("pretrained", exist_ok=True)
    acc_pred_checkpoint_path = (
        f"pretrained/{ofa_network.__class__.__name__}_acc_predictor.pth"
    )
    print(f"Accuracy Predictor Model. Loading checkpoint {acc_pred_checkpoint_path}. "
          f"Exists? {os.path.exists(acc_pred_checkpoint_path)}")

    acc_predictor = AccuracyPredictor(
        arch_encoder,
        hidden_size=400,
        n_layers=3,
        checkpoint_path=None,
        device1=device,
    )
    print(acc_predictor)

    # Accuracy Dataset
    print("Loading accuracy vs network arch feature map  dataset ...")
    acc_dataset = AccuracyDataset("acc_datasets")

    train_loader, valid_loader, base_acc = (
        acc_dataset.build_acc_data_loader(arch_encoder=arch_encoder, batch_size=256))

    print(f"The basic accuracy (mean accuracy of all subnets within the dataset is: {(base_acc * 100): .1f}%.")

    # Sample some architecture feature vectors
    sampled = 0
    for (data, label) in train_loader:
        data = data.to(device)
        label = label.to(device)
        print("=" * 100)
        # dummy pass to print the divided encoding
        arch_encoding = arch_encoder.feature2arch(data[0].int().cpu().numpy(), verbose=False)
        # print out the architecture encoding process in detail
        arch_encoding = arch_encoder.feature2arch(data[0].int().cpu().numpy(), verbose=True)
        visualize_subnet(arch_encoding)
        print(f"The accuracy of this subnet on the holdout validation set is: {(label[0] * 100): .1f}%.")
        sampled += 1
        if sampled == 1:
            break

    # ---------------------------------------------------------------------------------------
    # Training th Accuracy Predictor Model
    # ---------------------------------------------------------------------------------------
    print("Training the accuracy predictor model ...")

    criterion = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(acc_predictor.parameters())

    # the default value is zero
    acc_predictor.base_acc.data += base_acc

    for epoch in tqdm(range(10)):
        acc_predictor.train()
        for (data, label) in tqdm(train_loader, desc="Epoch%d" % (epoch + 1), position=0, leave=True):

            # step 1. Move the data and labels to device (cuda:0).
            data = data.to(device)
            label = label.to(device)

            # # step 2. Run forward pass.
            output = acc_predictor(data)  # output = [batch_size, 1]

            # # step 3. Calculate the loss.
            loss = criterion(output, label)

            # # step 4. Perform the backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_predictor.eval()
        with torch.no_grad():
            with tqdm(total=len(valid_loader), desc="Val", position=0, leave=True) as t:
                for (data, label) in valid_loader:

                    # step 1. Move the data and labels to device (cuda:0).
                    data = data.to(device)
                    label = label.to(device)

                    # step 2. Run forward pass.
                    output = acc_predictor(data)

                    # step 3. Calculate the loss.
                    loss = criterion(output, label)

                    t.set_postfix({"loss": loss.item()})
                    t.update(1)

    if not os.path.exists(acc_pred_checkpoint_path):
        torch.save(acc_predictor.cpu().state_dict(), acc_pred_checkpoint_path)

    # plot the correlation of predicted accuracy against ground truth accuracy and make sure our predictor is reliable.
    # -----------------------------------------------------------------------------------------------------------------
    print("Checking the accuracy of the accuracy predictor with ground truth values")
    predicted_accuracies = []
    ground_truth_accuracies = []

    acc_predictor = acc_predictor.to("cuda:0")
    acc_predictor.eval()

    with torch.no_grad():
        with tqdm(total=len(valid_loader), desc="Val") as t:
            for (data, label) in valid_loader:
                data = data.to(device)
                label = label.to(device)

                pred = acc_predictor(data)
                predicted_accuracies += pred.cpu().numpy().tolist()

                ground_truth_accuracies += label.cpu().numpy().tolist()

                if len(predicted_accuracies) > 200:
                    break

    plt.figure()
    plt.scatter(predicted_accuracies, ground_truth_accuracies)
    # draw y = x
    min_acc, max_acc = min(predicted_accuracies), max(predicted_accuracies)

    plt.plot([min_acc, max_acc], [min_acc, max_acc], c="red", linewidth=2, marker='+')

    plt.xlabel("Predicted accuracy")
    plt.ylabel("Measured accuracy")
    plt.title("Correlation between predicted accuracy and real accuracy")

    # -------------------------------------------------------------------------------------
    # NAS - Search
    # -------------------------------------------------------------------------------------
    print(f"Starting Random Search under constraints")
    nas_agent = RandomSearcher(efficiency_predictor, acc_predictor)

    # MACs-constrained search
    subnets_rs_macs = {}
    for millionMACs in [25, 100]:
        search_constraint = dict(millionMACs=millionMACs)
        print(f"Random search with constraint: MACs <= {millionMACs}M")
        subnets_rs_macs[millionMACs] = search_and_measure_acc(
            nas_agent, search_constraint, data_dir=data_directory, n_subnets=300)

    # memory-constrained search
    subnets_rs_memory = {}
    for KBPeakMemory in [128, 512]:
        search_constraint = dict(KBPeakMemory=KBPeakMemory)
        print(f"Random search with constraint: Peak memory <= {KBPeakMemory}KB")
        subnets_rs_memory[KBPeakMemory] = search_and_measure_acc(
            nas_agent, search_constraint, data_dir=data_directory, n_subnets=300)

    import pdb
    pdb.set_trace()
