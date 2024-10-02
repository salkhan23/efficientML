import os
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import namedtuple
from datetime import datetime

from fast_pytorch_kmeans import KMeans

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vgg import VGG  # noqa: E402  (ignore module import not at top)
import train_cifar10  # noqa: E402  (ignore module import not at top)
import model_analysis_utils  # noqa: E402  (ignore module import not at top)


Codebook = namedtuple('Codebook', ['centroids', 'labels'])


def k_means_quantize(fp32_tensor: torch.Tensor, bit_width=4, codebook=None):
    """
    Quantize tensor using k-means clustering.
    :param fp32_tensor: [torch.Tensor] Input tensor to quantize.
    :param bit_width: [int] Quantization bit width, default=4.
    :param codebook: [Codebook] Optional codebook for the centroids and labels.
    :return: [Codebook = (centroids, labels)]
             centroids: [torch.FloatTensor] The cluster centroids.
             labels: [torch.LongTensor] Cluster label tensor.
    """
    if codebook is None:
        # Get number of clusters based on the quantization precision
        n_clusters = 2 ** bit_width

        # Clustering algorithm does not fit on GPU for >8 bits, do it on CPU
        fp32_tensor_device = fp32_tensor.device
        fp32_tensor = fp32_tensor.to('cpu')

        # Use k-means to get the quantization centroids
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0, max_iter=300)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)

        fp32_tensor = fp32_tensor.to(fp32_tensor_device)

    # # Ensure codebook's centroids are on the same device as the tensor
    # codebook = Codebook(codebook.centroids.to(org_device), codebook.labels.to(org_device))

    # Decode the codebook into k-means quantized tensor for inference
    dists = torch.sqrt((torch.unsqueeze(fp32_tensor, -1) - codebook.centroids.to(fp32_tensor.device)) ** 2)
    labels = torch.argmin(dists, dim=-1).to(torch.long)

    centroids = codebook.centroids
    codebook = Codebook(centroids, labels)

    centroids = centroids.to(fp32_tensor.device)

    # Update the tensor with the quantized values
    quantized_tensor = centroids[labels].to(fp32_tensor.device)
    # fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    fp32_tensor.copy_(quantized_tensor.view_as(fp32_tensor))

    return codebook


def test_k_means_quantize(
    test_tensor=torch.tensor([
        [-0.3747,  0.0874,  0.3200, -0.4868,  0.4404],
        [-0.0402,  0.2322, -0.2024, -0.4986,  0.1814],
        [ 0.3102, -0.3942, -0.2030,  0.0883, -0.4741],
        [-0.1592, -0.0777, -0.3946, -0.2128,  0.2675],
        [ 0.0611, -0.1933, -0.4350,  0.2928, -0.1087]]),
        bit_width=2):
    """

    :param test_tensor:
    :param bit_width:
    :return:
    """

    def plot_matrix(tensor, ax, title, cmap=ListedColormap(['white'])):
        ax.imshow(tensor.cpu().numpy(), vmin=-4, vmax=4, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                ax.text(j, i, f'{tensor[i, j].item():.2f}', ha="center", va="center", color="k")

    fig, axes = plt.subplots(1, 2, figsize=(8, 12))
    ax_left, ax_right = axes.ravel()

    print(test_tensor)
    plot_matrix(test_tensor, ax_left, 'original tensor')

    num_unique_values_before_quantization = test_tensor.unique().numel()
    k_means_quantize(test_tensor, bit_width=bit_width)
    num_unique_values_after_quantization = test_tensor.unique().numel()
    print('* Test k_means_quantize()')
    print(f'    target bit-width: {bit_width} bits')
    print(f'        num unique values before k-means quantization: {num_unique_values_before_quantization}')
    print(f'        num unique values after  k-means quantization: {num_unique_values_after_quantization}')
    assert num_unique_values_after_quantization == min((1 << bit_width), num_unique_values_before_quantization)
    print('* Test passed.')

    plot_matrix(test_tensor, ax_right, f'{bit_width}-bit k-means quantized tensor', cmap='tab20c')
    fig.tight_layout()
    plt.show()


def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    """
    Update the centroids in the codebook using updated fp32_tensor
    :param fp32_tensor: [torch.(cuda.)Tensor]
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    """
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    new_centroids = codebook.centroids.clone()

    for k in range(n_clusters):
        cluster_points = fp32_tensor[codebook.labels.view(-1) == k]
        if cluster_points.numel() > 0:
            new_centroids[k] = cluster_points.mean()

    # Create a new codebook with updated centroids
    codebook = Codebook(new_centroids, codebook.labels)
    return codebook


class KMeansQuantizer:
    def __init__(self, model, bit_width=4):
        """"""
        self.codebook = self.quantize(model, bit_width)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        """ update centroids and then apply quantization """
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(param, codebook=self.codebook[name])

    @staticmethod
    @torch.no_grad()
    def quantize(model, bit_width=4):
        """"""
        codebook = dict()
        if isinstance(bit_width, dict):
            for name, param in model.named_parameters():
                if name in bit_width:
                    codebook[name] = k_means_quantize(param, bit_width=bit_width[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bit_width=bit_width)
        return codebook


if __name__ == "__main__":
    plt.ion()
    random_seed = 10
    torch.random.manual_seed(random_seed)

    # -----------------------------------------------------------------------------------
    # Test Quantization
    # -----------------------------------------------------------------------------------
    # test_k_means_quantize()

    # -----------------------------------------------------------------------------------
    # Quantize a whole model
    # -----------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_file = "../results_trained_models/07-06-2024_VGG_net_train_epochs_100_acc_81.pth"

    net = VGG()
    net.load_state_dict(torch.load(saved_model_file))
    net.to(device)

    b_size = 128
    data_dir = './data'

    train_data_set, train_loader, test_data_set, test_loader, _ = (
        train_cifar10.get_cifar10_datasets_and_data_loaders(data_dir=data_dir, b_size=b_size))

    # fp_model_acc = train_cifar10.evaluate(net, test_loader, device)
    # print(f"Floating point model accuracy {fp_model_acc:0.2f}%")
    #
    # n_bits = 2
    #
    # net = net.to('cpu')
    # print(f"{n_bits}-bit quantizing model ...")
    # quantizer = KMeansQuantizer(net, n_bits)
    # net = net.to('cuda')
    #
    # quant_model_acc = train_cifar10.evaluate(net, test_loader, device)
    # print(f"{n_bits}-bit quantized model accuracy (without fine-tuning) {quant_model_acc:0.2f}")

    # -----------------------------------------------------------------------------------
    # Fine-Tune quantized model
    #  To update centroids after each gradient step,
    #  Get gradient updated weights, then find the average of each cluster and update its centroid
    #  dL/dC_k = sum_over_j dL/dW.dW/dC_k = dl/dW. 1(Ij =c_k )
    #  where L is the loss function, Ck is k - th centroid, Ij is the label for weight Wj.
    #  1() is the indicator function, and 1(Ij=k) means 1 if Ij = k else 0, i.e., Ij == k.
    # -----------------------------------------------------------------------------------
    quantization_sizes = [2]

    accuracy_drop_threshold = 0.5
    quantizers_dict = {}

    best_acc = 0

    for q_size in quantization_sizes:
        """"""
        # Restore the original weights of the model
        net.load_state_dict(torch.load(saved_model_file))

        fp_model_acc = train_cifar10.evaluate(net, test_loader, device)
        print(f"Floating point model accuracy {fp_model_acc:0.2f}")

        print(f'k-means quantizing model into {q_size} bits')
        quantization_start_time = datetime.now()
        quantizer = KMeansQuantizer(net, q_size)  # Store the index labels and the cluster centroids.
        quantizer.apply(net, update_centroids=False)
        print(f"print: Initial quantization took: {datetime.now() - quantization_start_time}")
        # Todo: Figure out why just initializing, does not quantify the model, but calling apply does.

        start_quant_acc = train_cifar10.evaluate(net, test_loader, device)
        print(f"{q_size}-bit quantized model accuracy (without fine-tuning) {start_quant_acc:0.2f}")

        n_epochs = 10

        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(n_epochs):

            train_cifar10.train(
                net, train_loader, criterion, optimizer, scheduler,
                callbacks=[lambda: quantizer.apply(net, update_centroids=True)])
            # The lambda captures references to 'net' and 'update_centroids' instead of fixed values,
            # allowing it to be called without parameters inside 'train' while using the latest states.
            # This simplifies the callback mechanism as 'train' doesn't need to pass arguments explicitly.

            quant_acc = train_cifar10.evaluate(net, test_loader, device)
            print(f"Epoch{epoch + 1} fine-tuning acc = {quant_acc:0.2f}, lr={scheduler.get_last_lr()}")

            if quant_acc > best_acc:
                best_acc = quant_acc

            if quant_acc > (fp_model_acc + accuracy_drop_threshold):
                print(f"Reached floating point model accuracy. Exiting {quant_acc:0.2f}")
                break

        quantizers_dict[q_size] = quantizer
        # The model's weights don't need to be stored anymore; we can save just the quantizers instead.
        # To restore the model, simply load it and apply the quantizers without modifying the centroids.

    # -----------------------------------------------------------------------------------
    # Load a saved quantization and evaluate it
    # -----------------------------------------------------------------------------------

    import pdb
    pdb.set_trace()
