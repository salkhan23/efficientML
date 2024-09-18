import torch
from fast_pytorch_kmeans import KMeans


def k_means_cluster(fp32_tensor: torch.Tensor, k, max_iter=100, tol=1e-4):
    """
    k-means clustering on a floating-point tensor.

    :param fp32_tensor: floating point tensor to quantize
    :param k: Number of clusters
    :param max_iter: maximum number of iteration to find cluster centers
    :param tol:
    :return: centroids
    """

    tensor_flat = fp32_tensor.flatten()
    num_el = len(tensor_flat)

    # Randomly select k starting cluster centers
    centroids_idxs = torch.randint(0, num_el, (k,))
    centroids = tensor_flat[centroids_idxs]
    centroids = torch.sort(centroids)[0]
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

        print(f"{iter_idx}: centroids {centroids}. Dist with Prev: {torch.abs(prev_centroids - centroids).sum()}")

        if torch.abs(prev_centroids - centroids).sum() < 1e-4:
            print(f"Converged after {iter_idx + 1} iterations")
            break

        prev_centroids = centroids.clone()

    centroids = torch.sort(centroids)[0]
    print(f"Final Centroids; {centroids}")

    return centroids
#
#
# def k_means_quantize(fp32_tensor: torch.Tensor, codebook=None,  bit_width=4):
#     """
#     quantize tensor using k-means clustering
#
#     :param fp32_tensor:
#     :param bit_width: [int] quantization bit width, default=4
#     :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
#     :return:
#         [Codebook = (centroids, labels)]
#             centroids: [torch.(cuda.)FloatTensor] the cluster centroids
#             labels: [torch.(cuda.)LongTensor] cluster label tensor
#     """
#     if codebook is None:
#         # Create a new codebook with centroids and their labels
#         n_levels = 2**bit_width
#
#         codebook = k_means_cluster(fp32_tensor, n_levels)


if __name__ == "__main__":
    random_seed = 10
    torch.random.manual_seed(random_seed)

    test_tensor = torch.tensor([
        [-0.3747,  0.0874,  0.3200, -0.4868,  0.4404],
        [-0.0402,  0.2322, -0.2024, -0.4986,  0.1814],
        [ 0.3102, -0.3942, -0.2030,  0.0883, -0.4741],
        [-0.1592, -0.0777, -0.3946, -0.2128,  0.2675],
        [ 0.0611, -0.1933, -0.4350,  0.2928, -0.1087]])

    bit_width = 2
    n_clusters = 2**bit_width

    cluster_centers = k_means_cluster(test_tensor, k=n_clusters)
    print(f"Centroids Manual        : {cluster_centers}")

    # Use library function
    kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
    labels = kmeans.fit_predict(test_tensor.view(-1, 1)).to(torch.long)
    cluster_centers = kmeans.centroids.to(torch.float).view(-1)
    cluster_centers = torch.sort(cluster_centers)[0]

    print(f"Centroids Using Library : {cluster_centers}")



    import pdb
    pdb.set_trace()

