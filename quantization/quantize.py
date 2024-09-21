import os
import sys
import torch
from fast_pytorch_kmeans import KMeans

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vgg import VGG  # noqa: E402  (ignore module import not at top)
import train_cifar10  # noqa: E402  (ignore module import not at top)


def k_means_cluster(fp32_tensor: torch.Tensor, k, max_iter=100, tol=1e-4, verbose=False):
    """
    k-means clustering on a floating-point tensor.


    :param fp32_tensor: floating point tensor to quantize
    :param k: Number of clusters
    :param max_iter: maximum number of iteration to find cluster centers
    :param tol:
    :param verbose:
    :return: centroids
    """

    tensor_flat = fp32_tensor.flatten()
    num_el = len(tensor_flat)

    # Randomly select k starting cluster centers
    centroids_idxs = torch.randint(0, num_el, (k,))
    centroids = tensor_flat[centroids_idxs]
    centroids = torch.sort(centroids)[0]
    if verbose:
        print(f"Starting Centroids: {centroids}")

    prev_centroids = centroids.clone()

    # Expand dim by 1, allow broadcasting to do element wise subtraction
    for iter_idx in range(max_iter):
        dist = abs(tensor_flat.unsqueeze(-1) - centroids)
        closest_centroids = torch.argmin(dist, dim=-1)

        group_sums = torch.zeros_like(centroids)
        group_counts = torch.zeros_like(centroids)
        for i, centroid_idx in enumerate(closest_centroids):

            group_sums[centroid_idx] += tensor_flat[i]
            group_counts[centroid_idx] += 1

        # Update centroids by the mean of each cluster
        for g_idx in range(len(centroids)):
            if group_counts[g_idx] != 0:
                centroids[g_idx] = group_sums[g_idx] / group_counts[g_idx]

        # if verbose:
        print(f"{iter_idx}: centroids {centroids}. Dist with Prev: {torch.abs(prev_centroids - centroids).sum()}")

        if torch.abs(prev_centroids - centroids).sum() < tol:
            print(f"Converged after {iter_idx + 1} iterations")
            break

        prev_centroids = centroids.clone()

    centroids = torch.sort(centroids)[0]
    if verbose:
        print(f"Final Centroids; {centroids}")

    return centroids


def k_means_quantize(fp32_tensor: torch.Tensor, centroids, inplace=True):
    """
    quantize tensor using k-means clustering

    :param centroids:
    :param fp32_tensor:
    :param inplace:

    :return: quantized tensor
        [Codebook = (centroids, labels)]
            centroids: [torch.(cuda.)FloatTensor] the cluster centroids
            labels: [torch.(cuda.)LongTensor] cluster label tensor
    """
    # Expand dim by 1, allows broadcasting to do element-wise subtraction
    temp_tensor = torch.abs(fp32_tensor.unsqueeze(-1) - centroids)
    quantized_tensor = torch.argmin(temp_tensor, dim=-1)

    if inplace:
        fp32_tensor.copy_(quantized_tensor)

    return quantized_tensor


def k_means_cluster_fast(fp32_tensor, bit_width):
    """ Fast Kmeans Cluster using fast_pytorch_kmeans library"""
    num_clusters = 2 ** bit_width
    kmeans1 = KMeans(n_clusters=num_clusters, mode='euclidean', verbose=0)
    w_q_flat = kmeans1.fit_predict(fp32_tensor.view(-1, 1))

    # replace w with quantized w and store the centroids in a codebook
    fp32_tensor.copy_(w_q_flat.view_as(fp32_tensor))
    # Store centroids in codebook dictionary

    return kmeans1.centroids


class KMeansModelQuantizer:
    def __init__(self, model, bit_width=4):
        self.codebook = self.quantize(model, bit_width)

    @staticmethod
    @torch.no_grad()
    def quantize(model, bit_width):

        codebook = {}  # dictionary of param name = centroids

        if isinstance(bit_width, dict):  # Quantize named layers with specified bit width
            for name, param in model.named_parameters():
                if name in bit_width:
                    print(f"Quantizing {name}")
                    codebook[name] = k_means_cluster_fast(param, bit_width[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    print(f"Quantizing {name}")
                    codebook[name] = k_means_cluster_fast(param, bit_width)

        return codebook


if __name__ == "__main__":
    random_seed = 10
    torch.random.manual_seed(random_seed)

    # ---------------------------------------------------------------------------------------
    # Test K means clustering & quantizing
    # ---------------------------------------------------------------------------------------
    test_tensor = torch.tensor([
        [-0.3747,  0.0874,  0.3200, -0.4868,  0.4404],
        [-0.0402,  0.2322, -0.2024, -0.4986,  0.1814],
        [ 0.3102, -0.3942, -0.2030,  0.0883, -0.4741],
        [-0.1592, -0.0777, -0.3946, -0.2128,  0.2675],
        [ 0.0611, -0.1933, -0.4350,  0.2928, -0.1087]])

    n_bits = 2
    n_clusters = 2**n_bits

    cluster_centers = k_means_cluster(test_tensor, k=n_clusters)
    quantized_test_tensor = k_means_quantize(test_tensor, cluster_centers, inplace=False)
    print(f"Centroids Manual        : {cluster_centers}")
    print(quantized_test_tensor)

    # Use library function
    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
    kmeans.fit_predict(test_tensor.view(-1, 1))

    quantized_test_tensor = kmeans.predict(torch.reshape(test_tensor, (-1, 1)))
    quantized_test_tensor = quantized_test_tensor.view_as(test_tensor)
    print(f"Centroids Using Library  : {torch.reshape(kmeans.centroids, (1, -1))}")
    print(f"Note that the library does not use ordered centroids (quantized indexes may not match)")
    print(quantized_test_tensor)

    # ---------------------------------------------------------------------------------------
    # Quantize full model
    # ---------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_file = "../results_trained_models/07-06-2024_VGG_net_train_epochs_100_acc_81.pth"

    net = VGG()
    net.load_state_dict(torch.load(saved_model_file))
    net.to(device)

    b_size = 128
    data_dir = './data'

    train_data_set, train_loader, test_data_set, test_loader, _ = (
        train_cifar10.get_cifar10_datasets_and_data_loaders(data_dir=data_dir, b_size=b_size))

    model_acc = train_cifar10.evaluate(net, test_loader, device)
    print(f"Floating point Model  accuracy {model_acc:0.2f}")

    quantization_sizes = [2, 4, 8]

    for q_size in quantization_sizes:
        net = net.to('cpu')
        KMeansModelQuantizer(net, q_size)
        net = net.to('cuda')

        model_acc = train_cifar10.evaluate(net, test_loader, device)
        print(f"{q_size}-bit quantized Model  accuracy {model_acc:0.2f}")

        net.load_state_dict(torch.load(saved_model_file))

    import pdb
    pdb.set_trace()
