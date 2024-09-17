import torch


def k_means_cluster(fp32_tensor: torch.Tensor, k, max_iter=100):

    tensor_flat = fp32_tensor.flatten()
    num_el = len(tensor_flat)

    prob = torch.tensor([1/num_el]*num_el, dtype=torch.float)

    centroids_idxs = torch.multinomial(prob, k, replacement=False)
    centroids = tensor_flat[centroids_idxs]
    centroids = torch.sort(centroids)[0]

    tensor_flat = tensor_flat.unsqueeze(-1)  # Expand dim by 1, allow broadcasting to do element wise subtraction
    for iter_idx in range(max_iter):
        dist = abs(tensor_flat - centroids)
        closest_centroids = torch.argmin(dist, dim=-1)

        group_sums = torch.zeros_like(centroids)
        group_counts = torch.zeros_like(centroids)
        for i, centroid_idx in enumerate(closest_centroids):

            group_sums[centroid_idx] += tensor_flat[i,0]
            group_counts[centroid_idx] += 1

        # Update centroids by the mean of each cluster
        for g_idx in range(len(centroids)):
            if group_counts[g_idx] != 0:
                centroids[g_idx] = group_sums[g_idx] / group_counts[g_idx]

        print(centroids)


    import pdb
    pdb.set_trace()

    return 0


def k_means_quantize(fp32_tensor: torch.Tensor, codebook=None,  bit_width=4):
    """
    quantize tensor using k-means clustering

    :param fp32_tensor:
    :param bit_width: [int] quantization bit width, default=4
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    :return:
        [Codebook = (centroids, labels)]
            centroids: [torch.(cuda.)FloatTensor] the cluster centroids
            labels: [torch.(cuda.)LongTensor] cluster label tensor
    """
    if codebook is None:
        # Create a new codebook with centroids and their labels
        n_levels = 2**bit_width

        codebook = k_means_cluster(fp32_tensor, n_levels)

        import pdb
        pdb.set_trace()


if __name__ == "__main__":
    random_seed = 10
    torch.random.manual_seed(random_seed)

    test_tensor = torch.tensor([
        [-0.3747,  0.0874,  0.3200, -0.4868,  0.4404],
        [-0.0402,  0.2322, -0.2024, -0.4986,  0.1814],
        [ 0.3102, -0.3942, -0.2030,  0.0883, -0.4741],
        [-0.1592, -0.0777, -0.3946, -0.2128,  0.2675],
        [ 0.0611, -0.1933, -0.4350,  0.2928, -0.1087]])

    k_means_quantize(test_tensor,bit_width=2)





    import pdb
    pdb.set_trace()

