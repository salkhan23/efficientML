# -------------------------------------------------------------------------------------------------
# Linear Quantization of Model, weights and Activations
# -------------------------------------------------------------------------------------------------
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.colors import ListedColormap

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vgg import VGG  # noqa: E402  (ignore module import not at top)
import train_cifar10  # noqa: E402  (ignore module import not at top)
import model_analysis_utils  # noqa: E402  (ignore module import not at top)


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


def get_quantization_scale_and_zero_point(fp_tensor, bit_width):
    """
    S = (r_max - r_min) / (q_max - q_min)
    r_min = S(q_min - Z)  ==> Z = q_min - int(round(r_min/S))

    get quantization scale for single tensor

    :param fp_tensor: [torch.(cuda.)Tensor] floating tensor to be quantized
    :param bit_width: [int] quantization bit width
    :return:
        [float] scale
        [int] zero_point
    """
    quantized_min, quantized_max = get_quantized_range(bit_width)
    r_max = fp_tensor.max().item()
    r_min = fp_tensor.min().item()

    scale = (r_max - r_min)/(quantized_max - quantized_min)
    zero_point = quantized_min - round(r_min / scale)

    # clip the zero_point to fall in [quantized_min, quantized_max]
    zero_point = max(quantized_min, min(quantized_max, zero_point))

    return torch.tensor(scale), torch.tensor(zero_point, dtype=torch.int8)


def get_symmetric_linear_quantization_scale(fp_tensor, bit_width):
    """

    :param fp_tensor:
    :param bit_width:
    :return:
    """
    _, q_max = get_quantized_range(bit_width)

    r_max = max(fp_tensor.abs().max().item(), 5e-7)
    scale = r_max / q_max

    return torch.tensor(scale)


def linear_quantize_affine(fp_tensor, bit_width):
    """
    linear quantization fp_tensor using a single scale and zero-point.

    Function computes the scale and zero point and then calls linear_quantize to
    quantize fp_tensor with the specified bit_width

    :param fp_tensor: [torch.(cuda.)Tensor] floating feature to be quantized
    :param bit_width: [int] quantization bit width
    :return:
        [torch.(cuda.)Tensor] quantized tensor
        [float] scale tensor
        [int] zero point
    """
    scale, zero_point = get_quantization_scale_and_zero_point(fp_tensor, bit_width)
    quantized_tensor = linear_quantize(fp_tensor, bit_width, scale, zero_point)

    return quantized_tensor, scale, zero_point


def linear_quantize_symmetric_weight_per_channel(fp_tensor, bit_width):
    """
    Quantizes (per output channel) the given 4D fp_tensor using symmetric quantization.

    This function computes a symmetric quantization scale for each output channel,
    and then quantizes the entire tensor using those scales. The scale is reshaped
    to broadcast correctly over the tensor dimensions for efficient quantization.


    :param fp_tensor:  4D Tensor [ch_out, ch_in, r, c]
    :param bit_width:
    :return:
    """
    n_ch_out = fp_tensor.shape[0]

    scales_per_channel = torch.zeros(n_ch_out, device=fp_tensor.device)

    for ch_out_idx in range(n_ch_out):
        ch_out_tensor = fp_tensor[ch_out_idx, ]
        scales_per_channel[ch_out_idx] = get_symmetric_linear_quantization_scale(ch_out_tensor, bit_width)

    # modify scale so it is easily broadcast.
    # Same shape as fp_tensor, but with all but the output_channel dim=1.
    # Pytorch will automatically duplicate scale to the same shape as fp_tensor.
    scale_shape = [1] * fp_tensor.dim()
    scale_shape[0] = -1
    scales_per_channel = scales_per_channel.view(scale_shape)

    # now quantize fp_tensor
    quantized_tensor = linear_quantize(fp_tensor, bit_width, scales_per_channel, 0)

    return quantized_tensor, scales_per_channel, 0


def linear_quantize_symmetric_bias_per_output_channel(bias, w_scale, x_scale):
    """
    Perform per-channel symmetric quantization of the bias using weight and input activation scales.

    This function quantizes the bias tensor per output channel, using the product of weight scale (w_scale)
    and input activation scale (x_scale) as the bias scale. The zero point is fixed at 0, and a 32-bit
    quantization is applied to reduce quantization noise, given the sensitivity of bias to small errors.

    Assumptions:
    1. `bias_scale = w_scale * x_scale`
    2. Symmetric quantization with zero point = 0 for simpler computation of quantized outputs in layers like
       fully connected or convolutional.

    :param bias: 1D Tensor of bias values, matching the number of output channels.
    :param w_scale: Tensor or float representing the quantization scale for the weights.
    :param x_scale: Float representing the input activation scale.

    :return: Tuple (quantized_bias, bias_scale, zero_point), where quantized_bias is the quantized bias,
             bias_scale is the computed scale, and zero_point is always 0.
    """
    assert bias.dim() == 1, f"Expected bias to be 1D, but got {bias.dim()}D"
    assert bias.dtype == torch.float, f"Expected bias to be of type torch.float, but got {bias.dtype}"

    # All inputs use the same scale factor
    assert isinstance(x_scale, float), f"Expected input_scale to be a float, but got {type(x_scale)}"

    if isinstance(w_scale, torch.Tensor):
        assert w_scale.dtype == torch.float, f"Expected weight_scale to be of type torch.float, but got {w_scale.dtype}"

        # reshape the weight scale tensor (w_scale) into a 1D tensor (a vector) to ensure that it has
        # a shape of [n_output_channels], where n_output_channels is the number of output channels
        # or elements in the bias.
        w_scale = w_scale.view(-1)

        assert bias.numel() == w_scale.numel(), (
            f"Number of elements in bias ({bias.numel()}) does not match "
            f"number of elements in weight_scale ({w_scale.numel()})"
        )

    bias_scale = w_scale * x_scale

    quantized_bias = linear_quantize(bias, 32, bias_scale, zero_point=0, dtype=torch.int32)

    return quantized_bias, bias_scale, 0


def plot_weight_distribution(model, bit_width=32, extra_title=''):
    # bins = (1 << bit_width) if bit_width <= 8 else 256
    if bit_width <= 8:
        q_min, q_max = get_quantized_range(bit_width)
        bins = np.arange(q_min, q_max + 2)
        align = 'left'
    else:
        bins = 256
        align = 'mid'

    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()

    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            ax.hist(
                param.detach().view(-1).cpu(), bins=bins, density=True, align=align, color='blue', alpha=0.5,
                edgecolor='black' if bit_width <= 4 else None)

            if bit_width <= 4:
                quantized_min, quantized_max = get_quantized_range(bit_width)
                ax.set_xticks(np.arange(start=quantized_min, stop=quantized_max+1))

            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1

    fig.suptitle(f'Histogram of Weights (bit_width={bit_width} bits) {extra_title}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


@torch.no_grad()
def quantize_weights_and_plot_histogram(model, bit_width):
    """

    :param model:
    :param bit_width:
    :return:
    """

    for name, param in model.named_parameters():
        if param.dim() > 1:
            quantized_param, scale, zero_point = linear_quantize_symmetric_weight_per_channel(param, bit_width)
            param.copy_(quantized_param)

    plot_weight_distribution(model, bit_width, "Quantized")


if __name__ == "__main__":
    plt.ion()
    random_seed = 10
    torch.random.manual_seed(random_seed)

    # ---------------------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------------------
    # test_linear_quantize()

    # ---------------------------------------------------------------------------------------------
    # Quantize Weight matrices
    # ---------------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model_file = "../results_trained_models/07-06-2024_VGG_net_train_epochs_100_acc_81.pth"

    net = VGG()
    net.load_state_dict(torch.load(saved_model_file, weights_only=True))
    net.to(device)

    plot_weight_distribution(net, extra_title="Pre-quantization weight distribution")

    bit_widths = [2, 4, 8]

    for b_width in bit_widths:
        quantize_weights_and_plot_histogram(net, b_width)
        # Restore the model
        net.load_state_dict(torch.load(saved_model_file, weights_only=True))

    import pdb
    pdb.set_trace()
