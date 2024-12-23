# -------------------------------------------------------------------------------------------------
#  Linear Quantization of Neural networks
#
#  Affine (symmetric) quantization ofr input & output activations
#  Symmetric quantization for model weights.
#
#  Does:
#    [1] Quantizes a model for linear operation
#
#  Todo:
#   [1] Validation of convolutional layer
#   [2] Post-quantization fine tuning

#  Reference: https://colab.research.google.com/drive/11IBla1q1McoZ2oCANCGHns8VtzG5nCMP
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import os
import sys
from collections import namedtuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vgg import VGG   # noqa: E402  (ignore module import not at top)
import train_cifar10  # noqa: E402  (ignore module import not at top)


def get_quantize_range(bit_width):
    """ [2**(bit_width - 1), 2**(bit_width - 1) -1]. E.g. bit_width = 8: (-128, 127) """
    q_max = (1 << (bit_width - 1)) - 1
    q_min = -(1 << (bit_width - 1))
    return q_min, q_max


# -------------------------------------------------------------------------------------------------
#  Linear Quantization Functions
# -------------------------------------------------------------------------------------------------
def linear_quantize_with_scale_and_zero_point(
        fp_tensor,
        bit_width,
        scale: torch.Tensor | float,
        zero_point: torch.Tensor | int,
        dtype=torch.int8) -> torch.Tensor:
    """
    Quantize a given floating point tensor (fp_tensor) to bit_width bits, using the given scale and zero point

    The quantization process is defined as:
        r = S(q - Z)
        q = int(round(r/S)) + Z

    Where:
        r         : Input floating-point tensor (fp_tensor).
        S         : Scale factor.
        Z         : Zero point.
        q         : Quantized tensor output.

    Scale and zero_point can be single values or tensors, allowing for per-channel quantization.

    :param fp_tensor: Input tensor of floating-point values.
    :param bit_width: Number of bits for quantization (e.g., 8 for int8).
    :param scale: Scale factor, can be a float or a tensor matching the dimensions of fp_tensor.
    :param zero_point: Zero point, can be an int or a tensor matching the dimensions of fp_tensor.
    :param dtype: Desired data type for the quantized tensor (default: int8).
    :return: Quantized tensor with specified bit-width.
    """
    assert (fp_tensor.dtype == torch.float), \
        f"fp_tensor should have a data type of torch.float. Found {fp_tensor.dtype}"

    # Ensure scale and zero_point are of valid types and dimensions
    assert isinstance(scale, float) or (scale.dtype == torch.float and scale.dim() == fp_tensor.dim()), \
        "Scale must be a float or a tensor matching the dimensions of the input tensor."

    assert isinstance(zero_point, int) or (zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()), \
        "Zero point must be an int or a tensor matching the dimensions of the input tensor."

    scaled_tensor = fp_tensor / scale
    rounded_tensor = torch.round(scaled_tensor)

    shifted_tensor = rounded_tensor + zero_point

    # clip all elements to max, min range.
    # fp_tensor can contain values outside this quantization range and round may also push some out.
    q_min, q_max = get_quantize_range(bit_width)

    quantized_tensor = torch.clamp_(shifted_tensor, q_min, q_max)
    quantized_tensor = quantized_tensor.to(dtype)

    return quantized_tensor


def get_quantization_scale_and_zero_point(fp_tensor, bit_width):
    """
    Get quantization scale and zero point for single tensor (general, no assumptions method)

    S = (r_max - r_min) / (q_max - q_min)
    Z = q_min - int(round(r_min/S))

    r_max, r_min of the fp_tensor are calculated internally.

    :param fp_tensor: [torch.(cuda.)Tensor] floating tensor to be quantized
    :param bit_width: [int] quantization bit width

    :return: scale (float), zero_point (int)
    """
    q_min, q_max = get_quantize_range(bit_width)

    r_max = fp_tensor.max().item()
    r_min = fp_tensor.min().item()

    # Scale
    scale = (r_max - r_min) / (q_max - q_min)

    # Zero point
    zero_point = q_min - (r_min / scale)
    zero_point = int(round(zero_point))

    # Clamp zero_point to q_max, q_min range
    zero_point = min(max(zero_point, q_min), q_max)

    return scale, zero_point


def linear_quantize_affine(fp_tensor, bit_width):
    """
    linear quantize fp-tensor (general no assumptions)

    Calculates scale and zero point of the fp_tensor, then quantizes it.

    :param fp_tensor: [torch.(cuda.)Tensor] floating feature to be quantized
    :param bit_width: [int] quantization bit width

    :return: quantized tensor ([torch.(cuda.)Tensor] ), scale tensor (float), zero point (int)
    """
    scale, zero_point = get_quantization_scale_and_zero_point(fp_tensor, bit_width)

    quantized_tensor = linear_quantize_with_scale_and_zero_point(fp_tensor, bit_width, scale, zero_point)

    return quantized_tensor, scale, zero_point


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

    q_min, q_max = get_quantize_range(bit_width)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    plot_matrix(test_tensor, axes[0], 'original tensor', vmin=real_min, vmax=real_max, cmap='bwr')

    _quantized_test_tensor = linear_quantize_with_scale_and_zero_point(
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
        vmin=q_min, vmax=q_max, cmap='bwr')

    plot_matrix(
        _reconstructed_test_tensor, axes[2], f'reconstructed tensor', vmin=real_min, vmax=real_max, cmap='bwr')

    fig.tight_layout()
    plt.show()


# -------------------------------------------------------------------------------------------------
#  Symmetric Linear Quantization Functions
# -------------------------------------------------------------------------------------------------
def get_symmetric_quantization_scale(fp_tensor, bit_width):
    """
    Get symmetric quantization scale for fp_tensor

        S = r_max / q_max
        Z = 0

    In symmetric quantization, r_max = - r_min = max(abs(fp_tensor))

    :param fp_tensor: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bit_width: [integer] quantization bit width

    :return: scale (float)
    """
    r_max = max(fp_tensor.abs().max().item(), 5e-7)

    _, q_max = get_quantize_range(bit_width)

    scale = r_max / q_max

    return scale


def linear_quantize_symmetric_weight_per_output_channel(fp_weight, bit_width):
    """
    Apply symmetric per (output) channel quantization to given 4D fp_tensor  [ch_out, ch_in, r, c].

        S = r_max / q_max
        Z = 0

        where:  r_max = - r_min = max(abs(fp_tensor))

    :param fp_weight:
    :param bit_width:

    :return: quantized tensor ([torch.(cuda.)Tensor] ), scale tensor (float), zero point (int)
    """
    n_ch_out = fp_weight.shape[0]

    scales_per_channel = torch.zeros(n_ch_out, device=fp_weight.device)

    for ch_out_idx in range(n_ch_out):
        ch_out_tensor = fp_weight[ch_out_idx,]
        scales_per_channel[ch_out_idx] = get_symmetric_quantization_scale(ch_out_tensor, bit_width)

    # Modify scale so it is easily broadcast.
    # Same shape as fp_tensor, but with all dimensions other than the output dimension = 1.
    # Pytorch will automatically duplicate scale to the same shape as fp_tensor (broadcasting)
    scale_shape = [1] * fp_weight.dim()
    scale_shape[0] = -1
    scales_per_channel = scales_per_channel.view(scale_shape)

    # now quantize fp_tensor
    quantized_weight = linear_quantize_with_scale_and_zero_point(
        fp_weight, bit_width, scales_per_channel, 0)

    return quantized_weight, scales_per_channel, 0


def linear_quantize_symmetric_bias_per_output_channel(fp_bias, w_scale, input_scale):
    """
    Apply symmetric quantization to the bias term for each output channel.

    This function scales the floating-point bias (`fp_bias`) using the product of the weight scale
    and the input scale. It assumes the zero point is zero.

    Part of the linear layer quantization process (see quantized_linear_layer).

    Quantization formulas:
        S = S_w * S_x  (scale)
        Z = 0          (zero point)

    :param fp_bias: [torch.FloatTensor] bias weight to be quantized
    :param w_scale: [float or torch.FloatTensor] weight scale tensor
    :param input_scale: [float] input scale

    :return: quantized tensor ([torch.(cuda.)Tensor] ), scale tensor (float), zero point (int)
    """
    assert fp_bias.dim() == 1, "Bias tensor must be 1D (one bias per output channel)."
    assert fp_bias.dtype == torch.float, "Bias tensor must be of type torch.float."
    assert isinstance(input_scale, float), "Input scale must be a float."

    if isinstance(w_scale, torch.Tensor):
        assert w_scale.dtype == torch.float, "Weight scale tensor must be of type torch.float."
        w_scale = w_scale.view(-1)  # Flatten to 1D
        assert fp_bias.numel() == w_scale.numel(), \
            f"Weight scale size ({w_scale.numel()}) must match the bias size ({fp_bias.numel()})."

    scale = w_scale * input_scale

    # Fix 32 bit quantization is used for the bias.
    quantized_bias = linear_quantize_with_scale_and_zero_point(
        fp_bias, 32, scale, 0, dtype=torch.int32)

    return quantized_bias, scale, 0


# -------------------------------------------------------------------------------------------------
#  Full Layer Linear Quantization
# -------------------------------------------------------------------------------------------------
def shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    Computes the pre-computable shifted bias for quantized linear layer
    where:

        shifted_bias = q_b - Z_x * sum(q_w)

        - q_b: quantized bias
        - Z_x: input zero-point
        - sum(q_w): sum of quantized weights for each output channel (assumes per-channel quantization)

    Part of the linear layer quantization process (see quantized_linear_layer).

    :param quantized_bias: [torch.IntTensor] Quantized bias (should be int32).
    :param quantized_weight: [torch.CharTensor] Quantized weights (should be int8).
    :param input_zero_point: [int] Input zero-point (int).

    :return: Shifted quantized bias tensor. [n_out_ch]
    """
    assert (quantized_bias.dtype == torch.int32), \
        f"Expected bias to be of type torch.float, but got {quantized_bias.dtype}"
    assert (isinstance(input_zero_point, int)), \
        f"Expected input zero point to be of type int, got {type(input_zero_point)}"

    # Sum all weights over all input channel weights of each output channel
    shifted_q_bias = quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point

    return shifted_q_bias


def quantized_linear_layer(
        input_x, weight, shifted_q_bias, feature_bit_width, weight_bit_width, input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale):
    """
    Quantized fully-connected (linear) layer.

    The quantized output is computed as:
        q_y = (Linear[q_x, q_w] + shifted_bias) * (S_x * S_w / S_y) + Z_y

    Where
        - shifted_bias = shifted quantized bias (see shift_quantized_linear_bias)
        - S_x = quantized input scale
        - S_w = quantized weight scales (see linear_quantize_symmetric_bias_per_output_channel)
        - S_y = quantized output scale
        - Z_y = Output zero-point

    :param input_x: quantized input (torch.int8)
    :param weight: quantized weight (torch.int8)
    :param shifted_q_bias: shifted quantized bias or None (torch.int32)
    :param feature_bit_width: quantization bit width of input and output (int)
    :param weight_bit_width: quantization bit width of weight (int)
    :param input_zero_point: input zero point (int)
    :param output_zero_point: output zero point (int)
    :param input_scale: input feature scale (float)
    :param weight_scale: weight per-channel scale (torch.FloatTensor)
    :param output_scale: output feature scale (float)

    :return: quantized output activation (torch.int8)
    """
    assert input_x.dtype == torch.int8, f"Expected input dtype torch.int8, got {input_x.dtype}"
    assert weight.dtype == input_x.dtype, f"Expected weight dtype to match input, got {weight.dtype}"
    assert shifted_q_bias is None or shifted_q_bias.dtype == torch.int32, \
        f"Expected bias dtype torch.int32, got {shifted_q_bias.dtype}"
    assert isinstance(input_zero_point, int), f"Expected input_zero_point to be int, got {type(input_zero_point)}"
    assert isinstance(output_zero_point, int), f"Expected output_zero_point to be int, got {type(output_zero_point)}"
    assert isinstance(input_scale, float), f"Expected input_scale to be float, got {type(input_scale)}"
    assert isinstance(output_scale, float), f"Expected output_scale to be float, got {type(output_scale)}"
    assert weight_scale.dtype == torch.float, f"Expected weight_scale to be float, got {weight_scale.dtype}"

    # Step 1: int matrix multiplication (8-bit multiplication with 32-bit accumulation) and add shifted bias
    if 'cpu' in input_x.device.type:
        # use 32-b MAC for simplicity
        output = torch.nn.functional.linear(input_x.to(torch.int32), weight.to(torch.int32), shifted_q_bias)
    else:
        # current version pytorch does not yet support integer-based linear() on GPUs
        output = torch.nn.functional.linear(input_x.float(), weight.float(), shifted_q_bias.float())

    # Step 2: scale the output
    # Weight_scale: [oc, 1, 1, 1] -> [oc], then expanded to [1, oc] to match output shape [batch_size, oc]
    weight_shape = [1] * output.dim()
    weight_shape[1] = -1
    weight_scale = weight_scale.view(weight_shape)
    output = output * input_scale * weight_scale / output_scale

    # Step 3: shift output by output_zero_point
    output = output + output_zero_point

    # Step 4: Clamp the output to the valid quantization range and cast to int8
    q_min, q_max = get_quantize_range(feature_bit_width)
    output = output.round().clamp(q_min, q_max).to(torch.int8)

    return output


def test_quantized_fc(
        input_x=torch.tensor([
            [0.6118, 0.7288, 0.8511, 0.2849, 0.8427, 0.7435, 0.4014, 0.2794],
            [0.3676, 0.2426, 0.1612, 0.7684, 0.6038, 0.0400, 0.2240, 0.4237],
            [0.6565, 0.6878, 0.4670, 0.3470, 0.2281, 0.8074, 0.0178, 0.3999],
            [0.1863, 0.3567, 0.6104, 0.0497, 0.0577, 0.2990, 0.6687, 0.8626]]),
        weight=torch.tensor([
            [ 1.2626e-01, -1.4752e-01,  8.1910e-02,  2.4982e-01, -1.0495e-01, -1.9227e-01, -1.8550e-01, -1.5700e-01],
            [ 2.7624e-01, -4.3835e-01,  5.1010e-02, -1.2020e-01, -2.0344e-01, 1.0202e-01, -2.0799e-01,  2.4112e-01],
            [-3.8216e-01, -2.8047e-01,  8.5238e-02, -4.2504e-01, -2.0952e-01, 3.2018e-01, -3.3619e-01,  2.0219e-01],
            [ 8.9233e-02, -1.0124e-01,  1.1467e-01,  2.0091e-01,  1.1438e-01, -4.2427e-01,  1.0178e-01, -3.0941e-04],
            [-1.8837e-02, -2.1256e-01, -4.5285e-01,  2.0949e-01, -3.8684e-01, -1.7100e-01, -4.5331e-01, -2.0433e-01],
            [-2.0038e-01, -5.3757e-02,  1.8997e-01, -3.6866e-01,  5.5484e-02, 1.5643e-01, -2.3538e-01,  2.1103e-01],
            [-2.6875e-01,  2.4984e-01, -2.3514e-01,  2.5527e-01,  2.0322e-01, 3.7675e-01,  6.1563e-02,  1.7201e-01],
            [ 3.3541e-01, -3.3555e-01, -4.3349e-01,  4.3043e-01, -2.0498e-01, -1.8366e-01, -9.1553e-02, -4.1168e-01]]),
        bias=torch.tensor([ 0.1954, -0.2756,  0.3113,  0.1149,  0.4274,  0.2429, -0.1721, -0.2502]),
        quantized_bias=torch.tensor([ 3, -2,  3,  1,  3,  2, -2, -2], dtype=torch.int32),
        shifted_quantized_bias=torch.tensor([-1,  0, -3, -1, -3,  0,  2, -4], dtype=torch.int32),
        calc_quantized_output=torch.tensor([
            [ 0, -1,  0, -1, -1,  0,  1, -2],
            [ 0,  0, -1,  0,  0,  0,  0, -1],
            [ 0,  0,  0, -1,  0,  0,  0, -1],
            [ 0,  0,  0,  0,  0,  1, -1, -2]], dtype=torch.int8),
        bit_width=2,
        batch_size=4,
        in_channels=8,
        out_channels=8):
    """

    :param input_x:
    :param weight:
    :param bias:
    :param quantized_bias:
    :param shifted_quantized_bias:
    :param calc_quantized_output:
    :param bit_width:
    :param batch_size:
    :param in_channels:
    :param out_channels:
    :return:
    """

    def plot_matrix(tensor, ax, title, vmin=0., vmax=1., cmap='bwr'):
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

    output = torch.nn.functional.linear(input_x, weight, bias)

    # Quantize Weight matrix
    quantized_weight, weight_scale, weight_zero_point = (
        linear_quantize_symmetric_weight_per_output_channel(weight, bit_width))

    # Quantize the input
    quantized_input, input_scale, input_zero_point = linear_quantize_affine(input_x, bit_width)

    # Quantize the bias
    _quantized_bias, bias_scale, bias_zero_point = \
        linear_quantize_symmetric_bias_per_output_channel(bias, weight_scale, input_scale)
    assert _quantized_bias.equal(_quantized_bias)  # check for NaNs

    # Calculate linear quantization output ----------------------------------------------------
    # shifted_quantized_bias part
    _shifted_quantized_bias = \
        shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point)
    assert _shifted_quantized_bias.equal(shifted_quantized_bias)  # check that values match with input param

    # Quantize the output
    quantized_output, output_scale, output_zero_point = linear_quantize_affine(output, bit_width)

    # Calculate the output of the fully connected layer
    _calc_quantized_output = quantized_linear_layer(
        quantized_input, quantized_weight, shifted_quantized_bias,
        bit_width, bit_width,
        input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale)
    assert _calc_quantized_output.equal(calc_quantized_output)  # check that values match with input param

    # Show the reconstructed output
    reconstructed_calc_output = output_scale * (calc_quantized_output.float() - output_zero_point)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    quantized_min, quantized_max = get_quantize_range(bit_width)

    plot_matrix(weight, axes[0, 0], 'original weight', vmin=-0.5, vmax=0.5)
    plot_matrix(input_x.t(), axes[1, 0], 'original input', vmin=0, vmax=1)
    plot_matrix(output.t(), axes[2, 0], 'original output', vmin=-1.5, vmax=1.5)

    plot_matrix(quantized_weight, axes[0, 1], f'{bit_width}-bit linear quantized weight',
                vmin=quantized_min, vmax=quantized_max, cmap='bwr')
    plot_matrix(quantized_input.t(), axes[1, 1], f'{bit_width}-bit linear quantized input',
                vmin=quantized_min, vmax=quantized_max, cmap='bwr')
    plot_matrix(calc_quantized_output.t(), axes[2, 1], f'quantized output from quantized_linear_layer()',
                vmin=quantized_min, vmax=quantized_max, cmap='bwr')

    # Show the reconstructed weights
    reconstructed_weight = weight_scale * (quantized_weight.float() - weight_zero_point)
    reconstructed_input = input_scale * (quantized_input.float() - input_zero_point)
    # reconstructed_bias = bias_scale * (quantized_bias.float() - bias_zero_point)

    plot_matrix(reconstructed_weight, axes[0, 2], f'reconstructed weight',
                vmin=-0.5, vmax=0.5, cmap='bwr')
    plot_matrix(reconstructed_input.t(), axes[1, 2], f'reconstructed input',
                vmin=0, vmax=1, cmap='bwr')
    plot_matrix(reconstructed_calc_output.t(), axes[2, 2], f'reconstructed output',
                vmin=-1.5, vmax=1.5, cmap='bwr')

    print('* Test quantized_fc()')
    print(f'    target bit_width: {bit_width} bits')
    print(f'      batch size: {batch_size}')
    print(f'      input channels: {in_channels}')
    print(f'      output channels: {out_channels}')
    print('* Test passed.')
    fig.tight_layout()
    plt.show()


# -------------------------------------------------------------------------------------------------
# Convolutional Layer quantization
# -------------------------------------------------------------------------------------------------
def shift_quantized_conv2d_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    Computes the pre-computable shifted bias for quantized conv layers
    where:
        shifted_bias = q_b - conv(q_w, Z_x)

        - q_b: quantized bias
        - Z_x: input zero-point
        - q_w: quantized weight [ch-out, ch_in, r, c]

    Part of the conv layer quantization process (see quantized_conv_layer)

    Note: This function does not perform an actual convolution. Instead, it sums the weights across
    the input channels and spatial dimensions, then multiplies by the input zero point of the channel
    which is a single number.

    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point

    :return: shifted quantized bias tensor [n_out_ch]
    """
    assert quantized_bias.dtype == torch.int32, "Expected quantized bias to be of type int32."
    assert isinstance(input_zero_point, int), "Input zero point must be an integer."

    return quantized_bias - quantized_weight.sum((1, 2, 3)).to(torch.int32) * input_zero_point


def quantized_conv2d_layer(
        input_x, weight, shifted_q_bias, feature_bit_width, weight_bit_width, input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale, stride, padding, dilation, groups):
    """
    Quantized 2d convolution

    q_y = (CONV[q_x,q_w] + shifted_bias)⋅(S_x * S_w / S_y) + Z_y

    :param groups:
    :param dilation:
    :param padding:
    :param stride:
    :param input_x: quantized input (torch.int8)
    :param weight: quantized weight (torch.int8)
    :param shifted_q_bias: shifted quantized bias or None (torch.int32)
    :param feature_bit_width: quantization bit width of input and output (int)
    :param weight_bit_width: quantization bit width of weight (int)
    :param input_zero_point: input zero point (int)
    :param output_zero_point: output zero point (int)
    :param input_scale: input feature scale (float)
    :param weight_scale:  weight per-channel scale (float tensor)
    :param output_scale: output feature scale (float)

    :return: quantized output feature (float tensor)
    """
    assert len(padding) == 4, "Padding must be a length 4 tuple."
    assert input_x.dtype == torch.int8, "Input tensor must be of type torch.int8."
    assert weight.dtype == torch.int8, "Weight tensor must be of type torch.int8."
    assert shifted_q_bias is None or shifted_q_bias.dtype == torch.int32, "Bias must be of type torch.int32 or None."
    assert isinstance(input_zero_point, int), "Input zero point must be an integer."
    assert isinstance(output_zero_point, int), "Output zero point must be an integer."
    assert isinstance(input_scale, float), "Input scale must be a float."
    assert isinstance(output_scale, float), "Output scale must be a float."
    assert weight_scale.dtype == torch.float, "Weight scale must be of type torch.float."

    # Step 1: Pad input tensor with input_zero_point
    # In quantized neural networks, the input tensor is quantized using a zero point that may be different from 0.
    # Padding with the same zero point ensures that the padded areas aligns with the quantization scheme,
    # Making the convolution operation more consistent across the entire input tensor.
    # It is handled here separately for greater control
    input_x = torch.nn.functional.pad(input_x, padding, 'constant', value=input_zero_point)

    # Step 2: calculate integer-based 2d convolution (8-bit multiplication with 32-bit accumulation)
    if 'cpu' in input_x.device.type:
        # use 32-b MAC for simplicity (What should be done is 8 bit convolution, stored in 32-bit number)
        # Torch doesn't natively support 8-bit integer (int8) convolution on the CPU. There are specialized
        # quantized modules for int8 operations (like torch.quantized.conv2d), but using regular conv2d
        # with int8 can lead to unsupported operations or inconsistent results.
        output = torch.nn.functional.conv2d(
            input_x.to(torch.int32), weight.to(torch.int32), None, stride, 0, dilation, groups)
    else:
        # current version pytorch does not yet support integer-based conv2d() on GPUs
        output = torch.nn.functional.conv2d(
            input_x.float(), weight.float(), None, stride, 0, dilation, groups)

        output = output.round().to(torch.int32)

    # Step 3: Add shifted bias if present
    if shifted_q_bias is not None:
        # Output tensor from a convolution is shaped as (batch_size, out_channels, height, width)
        bias_shape = [1] * output.dim()
        bias_shape[1] = -1
        output = output + shifted_q_bias.view(bias_shape)

    # Step 4: Scale the output
    weight_shape = [1] * output.dim()
    weight_shape[1] = -1

    weight_scale = weight_scale.view(weight_shape)
    output = output * input_scale * weight_scale / output_scale

    # Step 5: Shift output by output_zero_point
    output = output + output_zero_point

    # Step 6: Clamp values to the quantized range and convert to int8
    q_min, q_max = get_quantize_range(feature_bit_width)
    output = output.round().clamp(q_min, q_max).to(torch.int8)

    return output


def fuse_conv_bn_layers(conv_layer, bn_layer):
    """
    Fuses a BatchNorm2d layer into the preceding Conv2d layer.

    This operation merges the normalization BN layer into the weights and biases of preceding Conv2D layer,
    removing the need for a separate BN during inference.

    Furthermore, BN introduces floating-point operations. By combining the 2 layers, the entire
    operation can be quantized at once, improving performance and reducing model size.

    **How It Works**:
    The fusion modifies the convolution layer’s weights and biases to include the effects of the BN
    layer's learned parameters:

    - **BatchNorm Parameters**:
        - `mu`: running mean (computed during training, bn_layer.running_mean).
        - `sigma^2`: running variance (computed during training, bn_layer.running var).
        - `gamma`: learnable scale parameter (`bn.weight`).
        - `beta`: learnable shift parameter (`bn.bias`).

    The fused weights and bias are computed as:
        - `w' = w * gamma / sqrt(sigma^2 + eps)`
        - `b' = (b - mu) * gamma / sqrt(sigma^2 + eps) + beta`

    ** Reference **:
    Modified from: https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html

    :param conv_layer: torch.nn.Conv2d: The convolutional layer to fuse.
    :param bn_layer: torch.nn.BatchNorm2d: The BatchNorm layer to fuse.

    :return: torch.nn.Conv2d: The fused Conv2d layer, with updated weights and biases.
    """
    assert conv_layer.bias is None, "Conv2d layer must not have a bias before fusion."

    # Compute the scaling factor = gamma / sqrt(running_var + eps)
    scale_factor = bn_layer.weight.data / torch.sqrt(bn_layer.running_var.data + bn_layer.eps)

    # Update the convolution weights: w' = w * scale_factor
    conv_layer.weight.data = conv_layer.weight.data * scale_factor.reshape(-1, 1, 1, 1)  # Per output channel

    # Update the convolution bias: b' = (b - mu) * scale_factor + beta
    # If no bias is present in conv, initialize it
    conv_layer.bias = torch.nn.Parameter(bn_layer.bias.data - bn_layer.running_mean.data * scale_factor)

    return conv_layer


# -------------------------------------------------------------------------------------------------
# Activation min/max finding functions
# -------------------------------------------------------------------------------------------------
Range = namedtuple('Range', ['r_min', 'r_max'])


def add_range_recoder_hooks(model, x_range_store_dict, y_range_store_dict):
    """
    Add hooks to monitor activation min, max values for model inputs and outputs.

    :param model: The model to attach hooks to.
    :param x_range_store_dict: Dictionary to store min and max values of inputs.
    :param y_range_store_dict: Dictionary to store min and max values of outputs.
    :return: List of all hooks.
    """

    def _record_range(_, x, y, mod_name):
        """
        Record min and max values of inputs and outputs for a given module.

        :param _: The module for which the hook is called. (part of feedforward hook callback)
        :param x: Input to the module (tuple).
        :param y: Output from the module.
        :param mod_name: Name of the module.
        """
        x = x[0]  # Inputs are tuples; take the first element

        # Calculate min and max for input activations
        x_min = x.min().item()
        x_max = x.max().item()

        # Calculate min and max for output activations
        y_min = y.min().item()
        y_max = y.max().item()

        # Update input ranges
        if mod_name not in x_range_store_dict:
            x_range_store_dict[mod_name] = Range(x_min, x_max)
        else:
            x_range_store_dict[mod_name] = Range(
                min(x_min, x_range_store_dict[mod_name].r_min),
                max(x_max, x_range_store_dict[mod_name].r_max)
            )

        # Update output ranges
        if mod_name not in y_range_store_dict:
            y_range_store_dict[mod_name] = Range(y_min, y_max)
        else:
            y_range_store_dict[mod_name] = Range(
                min(y_min, y_range_store_dict[mod_name].r_min),
                max(y_max, y_range_store_dict[mod_name].r_max)
            )

    all_hooks = []

    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU)):
            # Define a custom hook function that captures the module_name
            def hook_function(mod, x, y, mod_name=module_name):
                """
                Hook function called during the forward pass of the module.
                This function captures the module's input and output values
                and passes them to the _record_range function along with
                the module name.

                When the forward hook is triggered:
                - `mod` is the module where the hook is attached.
                - `x` is the input to the module, a tuple.
                - `y` is the output from the module.
                - `module_name` is captured from the outer scope and
                  corresponds to the name of the module, ensuring that
                  the correct range values are recorded in the dictionaries.

                :param mod: The module for which the hook is called.
                :param x: Input to the module (tuple).
                :param y: Output from the module.
                :param mod_name: Name of the module.
                """
                _record_range(mod, x, y, mod_name)

            # Register the forward hook
            hook_handle = module.register_forward_hook(hook_function)

            all_hooks.append(hook_handle)

    return all_hooks


# -------------------------------------------------------------------------------------------------
# Quantization Layers
# -------------------------------------------------------------------------------------------------
class QuantizedConv2d(torch.nn.Module):
    def __init__(
            self, weight, bias, in_zp, out_zp, in_scale, w_scale, out_scale, stride, padding, dilation, groups,
            feature_bit_width=8, weight_bit_width=8):
        """
        Create a quantized Conv2d Layer. This stores all the parameters needed for quantized
        operations and calls the quantized_conv2d function for the actual computation.

        :param weight:
        :param bias:
        :param in_zp:
        :param out_zp:
        :param in_scale:
        :param w_scale:
        :param out_scale:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param feature_bit_width:
        :param weight_bit_width:

        return a torch layer that does quantized conv2d operations
        """

        super().__init__()

        #  PyTorch's current version does not support using IntTensor as nn.Parameter.
        # Since backpropagation is not needed, Store these values as buffers using register_buffer.
        # Buffers are tensors that do not require gradients and are not involved in backpropagation.
        self.register_buffer('weight', weight)  # quantized weight
        self.register_buffer('bias', bias)

        self.input_zero_point = in_zp
        self.output_zero_point = out_zp

        self.input_scale = in_scale
        self.register_buffer('weight_scale', w_scale)
        self.output_scale = out_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bit_width = feature_bit_width
        self.weight_bit_width = weight_bit_width

    def forward(self, x):
        return quantized_conv2d_layer(
            input_x=x,
            weight=self.weight,
            shifted_q_bias=self.bias,
            feature_bit_width=self.feature_bit_width,
            weight_bit_width=self.weight_bit_width,
            input_zero_point=self.input_zero_point,
            output_zero_point=self.output_zero_point,
            input_scale=self.input_scale,
            weight_scale=self.weight_scale,
            output_scale=self.output_scale,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


class QuantizedLinear(torch.nn.Module):
    def __init__(
            self, weight, bias, input_zero_point, output_zero_point, input_scale, weight_scale, output_scale,
            feature_bit_width=8, weight_bit_width=8):
        """
        Create a quantized Linear Layer. It stores the parameters needed for quantization
        and calls the `quantized_linear_layer` function for computation.

        :param weight:
        :param bias:
        :param input_zero_point:
        :param output_zero_point:
        :param input_scale:
        :param weight_scale:
        :param output_scale:
        :param feature_bit_width:
        :param weight_bit_width:
        """

        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.feature_bit_width = feature_bit_width
        self.weight_bit_width = weight_bit_width

    def forward(self, x):
        return quantized_linear_layer(
            input_x=x,
            weight=self.weight,
            shifted_q_bias=self.bias,
            feature_bit_width=self.feature_bit_width,
            weight_bit_width=self.weight_bit_width,
            input_zero_point=self.input_zero_point,
            output_zero_point=self.output_zero_point,
            input_scale=self.input_scale, weight_scale=self.weight_scale,
            output_scale=self.output_scale
        )


class QuantizedMaxPool2d(torch.nn.MaxPool2d):
    """
    A quantized version of the MaxPool2d layer. This layer applies max pooling in floating point
    precision and converts the result back to int8, which is suitable for quantized models.

    Since PyTorch currently does not support integer-based max pooling, the input is first
    cast to float, max pooling is applied in float, and the output is converted back to int8.
    """
    def forward(self, x):
        # current version PyTorch does not support integer-based MaxPool
        return super().forward(x.float()).to(torch.int8)


class QuantizedAvgPool2d(torch.nn.AvgPool2d):
    """
    A quantized version of the AvgPool2d layer. This layer applies average pooling in floating point
    precision, adjusts for quantization scale differences between input and output, and converts the result
    back to int8.
    """
    def forward(self, x):
        # current version PyTorch does not support integer-based AvgPool
        return super().forward(x.float()).to(torch.int8)


# -------------------------------------------------------------------------------------------------
# Debugging functions
# -------------------------------------------------------------------------------------------------
def plot_weight_distributions(model, bit_width=32, extra_title=''):
    """

    :param model:
    :param bit_width:
    :param extra_title:
    :return:
    """
    # bins = (1 << bit_width) if bit_width <= 8 else 256
    if bit_width <= 8:
        q_min, q_max = get_quantize_range(bit_width)
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
                q_min, q_max = get_quantize_range(bit_width)
                ax.set_xticks(np.arange(start=q_min, stop=q_max + 1))

            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1

    fig.suptitle(f'Histogram of Weights (bit_width={bit_width} bits) {extra_title}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)


@torch.no_grad()
def quantize_model_weights(model, bit_width):
    """

    :param model:
    :param bit_width:
    :return:
    """
    for name, param in model.named_parameters():
        if param.ndim > 1:
            quantized_param, scale, zero_point = linear_quantize_symmetric_weight_per_output_channel(param, bit_width)
            param.copy_(quantized_param)


if __name__ == "__main__":
    plt.ion()
    random_seed = 10
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model_file = "../results_trained_models/27-10-2024_VGG_net_train_epochs_100_acc_93.pth"

    net = VGG()

    net.load_state_dict(torch.load(saved_model_file, weights_only=True))
    net.to(device)

    b_size = 128
    data_dir = './data'

    train_data_set, train_loader, test_data_set, test_loader, _ = (
        train_cifar10.get_cifar10_datasets_and_data_loaders(data_dir=data_dir, b_size=b_size))

    # print(f"Floating point model accuracy {train_cifar10.evaluate(net, test_loader, device):0.2f}")

    # # ---------------------------------------------------------------------------------------------
    # # Test Quantization Function
    # # ---------------------------------------------------------------------------------------------
    # print("Test linear quantization floating point tensor ...")
    # test_linear_quantize()
    #
    # print("Test linear quantize Full connected layer ...")
    # test_quantized_fc()

    # # TODO: Add function to test convolutional layer output

    # # ---------------------------------------------------------------------------------------------
    # # Quantize Model Weights
    # # Note this modifies the models weights. Need to reload net after this
    # # ---------------------------------------------------------------------------------------------
    # print("Plotting weight distributions for different quantization levels")
    # # Distribution of weights of the floating point model
    # plot_weight_distributions(net, extra_title='Floating point model')
    #
    # # Quantize model weights and plot weight distributions
    # quantized_bit_widths = [8, 4, 2]
    #
    # for q_bit_width in quantized_bit_widths:
    #     quantize_model_weights(net, q_bit_width)
    #
    #     plot_weight_distributions(net, q_bit_width, extra_title="Quantized")
    #
    #     # quant_model_acc = train_cifar10.evaluate(net, test_loader, device)
    #     # print(f"{q_bit_width}-bit quantized model accuracy (without fine-tuning) {quant_model_acc:0.2f}")
    #
    #     # Restore the model
    #     net.load_state_dict(torch.load(saved_model_file, weights_only=True))

    # ---------------------------------------------------------------------------------------------
    # Optimization - Fuse Convolutional & BN layer for less inference compute
    # Note: Commonly done in quantized networks - BN is just a scaling and a shifting operation
    # ---------------------------------------------------------------------------------------------
    print("Fusing model conv and BN layers for inference efficiency ...")

    # Load original model
    net.load_state_dict(torch.load(saved_model_file, weights_only=True))
    print('\tBefore conv-bn fusion model backbone length', len(net.backbone))

    model_fused = copy.deepcopy(net)

    fused_backbone = []
    ptr = 0

    while ptr < len(model_fused.backbone):
        if isinstance(model_fused.backbone[ptr], torch.nn.Conv2d) and \
                isinstance(model_fused.backbone[ptr + 1], torch.nn.BatchNorm2d):

            fused_backbone.append(
                fuse_conv_bn_layers(model_fused.backbone[ptr], model_fused.backbone[ptr + 1]))

            ptr += 2

        else:
            fused_backbone.append(model_fused.backbone[ptr])
            ptr += 1

    model_fused.backbone = torch.nn.Sequential(*fused_backbone)
    print('\tAfter  conv-bn fusion model backbone length', len(model_fused.backbone))

    # sanity check, no BN anymore
    for m in model_fused.modules():
        assert not isinstance(m, torch.nn.BatchNorm2d)

    print("conv BN fused model")
    print(model_fused)

    # Accuracy should remain the same after fusion
    fused_acc = train_cifar10.evaluate(model_fused, test_loader, device)
    print(f'Accuracy of the fused model = {fused_acc:.2f}%')

    # ---------------------------------------------------------------------------------------------
    # Find r_min/r_max of for input and outputs quantization
    # Add forward hooks to store peak input/output r_max and r_max for each network layer
    # ---------------------------------------------------------------------------------------------
    print("Finding input/output r_max & r_min for all network layers ...")

    # Add pytorch forward hook to record the min max value of the activation
    input_activation_ranges = {}
    output_activation_ranges = {}

    model_hooks = add_range_recoder_hooks(model_fused, input_activation_ranges, output_activation_ranges)

    # Pass some same data
    n_samples = 5
    for s_idx in range(n_samples):
        sample_input = iter(train_loader).__next__()[0]  # just the input
        model_fused(sample_input.to(device))

    # remove hooks
    for h in model_hooks:
        h.remove()

    # print min max values for each layer:
    for l_idx, layer in enumerate(input_activation_ranges.keys()):
        i_r_min, i_r_max = input_activation_ranges[layer]
        o_r_min, o_r_max = output_activation_ranges[layer]

        layer_type = ''
        if 'backbone' in layer:
            layer_type = type(model_fused.backbone[l_idx]).__name__
        elif 'classifier' in layer:
            layer_type = type(model_fused.classifier).__name__

        print(
            f"{layer:<12} ({layer_type:<12}): "
            f"Input ({i_r_min:7.2f}, {i_r_max:7.2f}) "
            f"Output ({o_r_min:7.2f}, {o_r_max:7.2f})")

    # Note: The ReLU layer operates in-place, which is why its input range is between 0 and r_max instead of reflecting
    # the output range from the BatchNorm and convolution layers.

    # # Debug - check if Relu is in place
    # for name_module1, module1 in model_fused.named_modules():
    #     if isinstance(module1, torch.nn.ReLU):
    #         print(f"{name_module1}: inplace={module1.inplace}")

    # ----------------------------------------------------------------------------------------
    # Convert a fp model to its quantized version for inference.
    # ----------------------------------------------------------------------------------------
    f_bit_width = 8  # Feature bit-width
    w_bit_width = 8  # weight bit-width

    quantized_model = copy.deepcopy(model_fused)

    quantized_backbone = []

    # Create the quantized model,
    # create quantization layers and store all parameter including all pre-computable parts
    ptr = 0
    while ptr < len(quantized_model.backbone):

        if (isinstance(quantized_model.backbone[ptr], torch.nn.Conv2d) and
                isinstance(quantized_model.backbone[ptr + 1], torch.nn.ReLU)):

            conv = quantized_model.backbone[ptr]
            conv_name = f'backbone.{ptr}'

            relu = quantized_model.backbone[ptr + 1]
            relu_name = f'backbone.{ptr + 1}'

            # Get all parameters needed to create the conv layer  ---------------------------------------

            # Get input scale and zero point scales
            i_r_min, i_r_max = input_activation_ranges[conv_name]
            i_scale, i_zp = get_quantization_scale_and_zero_point(torch.tensor([i_r_min, i_r_max]), f_bit_width)

            # Get output scale and zero point
            # Note output quantization is done after the relu (not after the fused conv-BN).
            # 3 advantages of this approach: (1) input fed into the next layer is between (0, r-max).
            # There is no need to spread the quantization range over negative values as these are not used.
            # (2) In the actual range, we can have greater resolution. (3) We can get rid of the relu op
            # as it is included in the quantization process.
            o_r_min, o_r_max = output_activation_ranges[relu_name]
            o_scale, o_zp = get_quantization_scale_and_zero_point(torch.tensor([o_r_min, o_r_max]), f_bit_width)

            # Get quantized weights
            quantized_w, scale_w, zp_w = \
                linear_quantize_symmetric_weight_per_output_channel(conv.weight.data, w_bit_width)

            # Get quantized bias
            quantized_b, scale_b, zp_b = \
                linear_quantize_symmetric_bias_per_output_channel(conv.bias.data, scale_w, i_scale)

            # compute shifted_q_bias
            shifted_quantized_b = shift_quantized_conv2d_bias(quantized_b, quantized_w, i_zp)

            # Form the quantized convolutional layer  --------------------------------------------------
            quantized_conv = QuantizedConv2d(
                quantized_w, shifted_quantized_b, i_zp, o_zp, i_scale, scale_w, o_scale,
                conv.stride, conv.padding, conv.dilation, conv.groups,
                feature_bit_width=f_bit_width, weight_bit_width=w_bit_width)

            quantized_backbone.append(quantized_conv)

            ptr += 2

        # Handle Max pooling layers
        elif isinstance(quantized_model.backbone[ptr], torch.nn.MaxPool2d):
            quantized_backbone.append(QuantizedMaxPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride
            ))
            ptr += 1

        # Handle Average pooling layer
        elif isinstance(quantized_model.backbone[ptr], torch.nn.AvgPool2d):
            quantized_backbone.append(QuantizedAvgPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride
            ))
            ptr += 1

        else:
            raise NotImplementedError(type(quantized_model.backbone[ptr]))  # should not happen

    quantized_model.backbone = torch.nn.Sequential(*quantized_backbone)

    # Handle the Classifier
    fc_name = 'classifier'
    fc = net.classifier

    # input scale and zp
    i_fc_r_min, i_fc_r_max = input_activation_ranges[fc_name]
    i_fc_scale, i_fc_zp = get_quantization_scale_and_zero_point(torch.tensor([i_fc_r_min, i_fc_r_max]), f_bit_width)

    # output scale and zp
    o_fc_r_min, o_fc_r_max = output_activation_ranges[fc_name]
    o_fc_scale, o_fc_zp = get_quantization_scale_and_zero_point(torch.tensor([o_fc_r_min, o_fc_r_max]), f_bit_width)

    # quantize the weight
    quantized_fc_w, scale_fc_w, zp_fc_w = (
        linear_quantize_symmetric_weight_per_output_channel(fc.weight.data, w_bit_width))

    # quantize the bias
    quantized_fc_b, scale_fc_b, zp_fc_b = \
        linear_quantize_symmetric_bias_per_output_channel(fc.bias.data, scale_fc_w, i_fc_scale)

    # Get shifted_q_bias
    shifted_quantized_b = shift_quantized_linear_bias(quantized_fc_b, quantized_fc_w, i_fc_zp)

    # Attach layer to model
    quantized_model.classifier = QuantizedLinear(
        quantized_fc_w, shifted_quantized_b,
        i_fc_zp, o_fc_zp, i_fc_scale, scale_fc_w, o_fc_scale,
        feature_bit_width=f_bit_width, weight_bit_width=w_bit_width)

    # print the model
    print("Quantized Model")
    print(quantized_model)

    # ---------------------------------------------------------------------------------------------
    # Evaluate the quantized model
    # ---------------------------------------------------------------------------------------------
    def quantize_model_inputs(x):
        """
        Quantizes input data in range [-1,1] by scaling to a symmetric range of [-128, 127] and converting to int8.
        """
        x = (x + 1) * 128 - 128

        return x.clamp(-128, 127).to(torch.int8)


    extra_preprocess = [quantize_model_inputs]

    int8_model_accuracy = train_cifar10.evaluate(quantized_model, test_loader, device, extra_preprocess)

    print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")

    import pdb
    pdb.set_trace()
