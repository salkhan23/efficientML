# -------------------------------------------------------------------------------------------------
# Linear Quantization of Model, weights and Activations
# -------------------------------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_quantized_range(bit_width):
    """
    [2**(bit_width - 1), 2**(bit_width - 1) -1].
    bit_width=8: (-128, 127)
    """
    q_max = (1 << (bit_width - 1)) - 1
    q_min = -(1 << (bit_width - 1))
    return q_min, q_max


def linear_quantize(
        fp_tensor,
        bit_width,
        scale: torch.Tensor | float,
        zero_point: torch.Tensor | int,
        dtype=torch.int8) -> torch.Tensor:
    """
    Quantize a floating-point tensor (fp_tensor) to a specified bit-width using linear quantization.

    fp_tensor = (quantized_tensor - zero_point) * scale
    quantized_tensor = int(round(fp_tensor / scale)) + zero_point

    This function supports per-channel quantization by allowing vectors of scales and zero points,
    enabling different quantization parameters for each output channel.

    :param fp_tensor:
    :param bit_width:
    :param scale:
    :param zero_point:
    :param dtype:
    :return:
    """
    assert (fp_tensor.dtype == torch.float)
    # Ensure scale and zero_point are of valid types and dimensions
    assert isinstance(scale, float) or (scale.dtype == torch.float and scale.dim() == fp_tensor.dim()), \
        "Scale must be a float or a tensor matching the dimensions of the input tensor."

    assert isinstance(zero_point, int) or (zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()), \
        "Zero point must be an int or a tensor matching the dimensions of the input tensor."

    scaled_tensor = fp_tensor / scale

    # In-place rounding; scaled_tensor is modified and no longer accessible as originally defined
    # Note: torch.Tensor.round_() is used rather than torch.round(), because it allows in-place rounding
    rounded_tensor = torch.Tensor.round_(scaled_tensor)

    shifted_tensor = rounded_tensor + zero_point

    # clip all elements to max, min range
    q_min, q_max = get_quantized_range(bit_width)
    quantized_tensor = torch.clamp_(shifted_tensor, q_min, q_max)

    quantized_tensor = quantized_tensor.to(dtype)

    return quantized_tensor


def test_linear_quantize(
        test_tensor=torch.tensor([
            [ 0.0523,  0.6364, -0.0968, -0.0020,  0.1940],
            [ 0.7500,  0.5507,  0.6188, -0.1734,  0.4677],
            [-0.0669,  0.3836,  0.4297,  0.6267, -0.0695],
            [ 0.1536, -0.0038,  0.6075,  0.6817,  0.0601],
            [ 0.6446, -0.2500,  0.5376, -0.2226,  0.2333]]),
        quantized_test_tensor=torch.tensor([
            [-1,  1, -1, -1,  0],
            [ 1,  1,  1, -2,  0],
            [-1,  0,  0,  1, -1],
            [-1, -1,  1,  1, -1],
            [ 1, -2,  1, -2,  0]], dtype=torch.int8),
        real_min=-0.25,
        real_max=0.75,
        bit_width=2,
        scale=1/3,
        zero_point=-1):
    """

    :param test_tensor:
    :param quantized_test_tensor:
    :param real_min:
    :param real_max:
    :param bit_width:
    :param scale:
    :param zero_point:
    :return:
    """

    def plot_matrix(tensor, ax, title, vmin=0., vmax=1., cmap=None):
        ax.imshow(tensor.cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                datum = tensor[i, j].item()
                if isinstance(datum, float):
                    ax.text(j, i, f'{datum:.2f}', ha="center", va="center", color="k")
                else:
                    ax.text(j, i, f'{datum}', ha="center", va="center", color="k")

    q_min, q_max = get_quantized_range(bit_width)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    plot_matrix(test_tensor, axes[0], 'original tensor', vmin=real_min, vmax=real_max)

    _quantized_test_tensor = linear_quantize(
        test_tensor,
        bit_width=bit_width,
        scale=scale,
        zero_point=zero_point)

    _reconstructed_test_tensor = scale * (_quantized_test_tensor.float() - zero_point)

    print('* Test linear_quantize()')
    print(f'\ttarget bit_width: {bit_width} bits')
    print(f'\t\tscale: {scale}')
    print(f'\t\tzero point: {zero_point}')

    assert _quantized_test_tensor.equal(quantized_test_tensor)  # test parameter
    print('* Test passed.')

    plot_matrix(
        _quantized_test_tensor, axes[1], f'{bit_width}-bit linear quantized tensor',
        vmin=q_min, vmax=q_max, cmap='tab20c')

    plot_matrix(
        _reconstructed_test_tensor, axes[2], f'reconstructed tensor', vmin=real_min, vmax=real_max, cmap='tab20c')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plt.ion()
    random_seed = 10
    torch.random.manual_seed(random_seed)

    test_linear_quantize()

    import pdb
    pdb.set_trace()
