# -------------------------------------------------------------------------------------------------
# Linear Quantization of Neural Networks
# Affine Transformation for activations and Symmetric Quantization for weights and biases
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vgg import VGG  # noqa: E402  (ignore module import not at top)
import train_cifar10  # noqa: E402  (ignore module import not at top)


def linear_quantize(
        fp_tensor, bit_width, scale: torch.Tensor | float, zero_point: torch.Tensor | int,
        dtype=torch.int8) -> torch.Tensor:
    """
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
        f"floating point tensor should have a data type of torch.float.f Found {fp_tensor.dtype}"

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


def get_quantize_range(bit_width):
    """ [2**(bit_width - 1), 2**(bit_width - 1) -1]. E.g. bit_width = 8: (-128, 127) """
    q_max = (1 << (bit_width - 1)) - 1
    q_min = -(1 << (bit_width - 1))
    return q_min, q_max


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
    get quantization scale and zero point for single tensor

    S = (r_max - r_min) / (q_max - q_min)
    Z = q_min - int(round(r_min/S))

    :param fp_tensor: [torch.(cuda.)Tensor] floating tensor to be quantized
    :param bit_width: [int] quantization bit width

    :return:
        [float] scale
        [int] zero_point
    """
    q_min, q_max = get_quantize_range(bit_width)
    r_max = fp_tensor.max().item()
    r_min = fp_tensor.min().item()

    # Calculate scale
    if r_max == r_min:
        scale = 0.0  # Handle edge case where range is zero
        zero_point = q_min  # Set zero_point to q_min

    else:
        scale = (r_max - r_min) / (q_max - q_min)

        zero_point = q_min - (r_min / scale)
        zero_point = int(round(zero_point))

    # Clamp zero_point to q_max, q_min range
    zero_point = min(max(zero_point, q_min), q_max)

    return scale, zero_point


def linear_quantize_affine(fp_tensor, bit_width):
    """
    linear quantization to given fp-tensor.

    Most general form of linear quantization - Calculates scale and zero point using max, min
    of fp_tensor

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


def get_symmetric_quantization_scale(fp_tensor, bit_width):
    """
    get symmetric quantization scale for fp_tensor

    In symmetric quantization
        (1) r_max = - r_min = max(abs(fp_tensor))
        (2) zero-point = 0

    S = r_max / q_max
    Z = 0

    :param fp_tensor: [torch.(cuda.)Tensor] floating weight to be quantized
    :param bit_width: [integer] quantization bit width

    :return:
        [floating scalar] scale
    """
    r_max = max(fp_tensor.abs().max().item(), 5e-7)

    _, q_max = get_quantize_range(bit_width)

    scale = r_max / q_max

    return scale


def linear_quantize_symmetric_weight_per_output_channel(fp_weight, bit_width):
    """
    Apply symmetric per_channel (output channels) quantization to given 4D fp_tensor  [ch_out, ch_in, r, c].

    In symmetric quantization
        (1) r_max = - r_min = max(abs(fp_tensor))
        (2) zero-point = 0

    S = r_max / q_max
    Z = 0

    :param fp_weight:
    :param bit_width:
    :return:
    """
    n_ch_out = fp_weight.shape[0]

    scales_per_channel = torch.zeros(n_ch_out, device=fp_weight.device)

    for ch_out_idx in range(n_ch_out):
        ch_out_tensor = fp_weight[ch_out_idx,]
        scales_per_channel[ch_out_idx] = get_symmetric_quantization_scale(ch_out_tensor, bit_width)

    # Modify scale so it is easily broadcast.
    # Same shape as fp_tensor, but with all dimensions other than the output dimension = 1.
    # Pytorch will automatically duplicate scale to the same shape as fp_tensor according to broadcasting rules
    scale_shape = [1] * fp_weight.dim()
    scale_shape[0] = -1
    scales_per_channel = scales_per_channel.view(scale_shape)

    # now quantize fp_tensor
    quantized_weight = linear_quantize(fp_weight, bit_width, scales_per_channel, 0)

    return quantized_weight, scales_per_channel, 0


def linear_quantize_symmetric_bias_per_output_channel(fp_bias, w_scale, input_scale):
    """
    Apply symmetric quantization for bias term.

    Note: this does not calculate its own bias, to simplify network layer output quantization,
    the scale is assumed to be weight_scale * input Scale

    S = S_w * S_x
    Z = 0

    :param fp_bias: [torch.FloatTensor] bias weight to be quantized
    :param w_scale: [float or torch.FloatTensor] weight scale tensor
    :param input_scale: [float] input scale
    :return:
        [torch.IntTensor] quantized bias tensor
    """
    assert fp_bias.dim() == 1, "Bias tensor must be 1D (one bias per output channel)."
    assert fp_bias.dtype == torch.float, "Bias tensor must be of type torch.float."
    assert isinstance(input_scale, float), "Input scale must be a float."

    if isinstance(w_scale, torch.Tensor):
        assert w_scale.dtype == torch.float, "Weight scale tensor must be of type torch.float."
        w_scale = w_scale.view(-1)  # Flatten to 1D if needed
        assert fp_bias.numel() == w_scale.numel(), \
            f"Weight scale size ({w_scale.numel()}) must match the bias size ({fp_bias.numel()})."

    scale = w_scale * input_scale

    quantized_bias = linear_quantize(fp_bias, 32, scale, 0, dtype=torch.int32)

    return quantized_bias, scale, 0


def shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    Computes the shifted bias for quantized linear layers (Q_bias).

    In quantized inference, the output of a dense (fully-connected) layer is represented as:

    q_y = (S_x * S_w / S_y) * (q_w * q_x + q_b - Z_x * sum(q_w)) + Z_y

    This function calculates the pre-computable portion of the equation:

    shifted_bias = q_b - Z_x * sum(q_w)

    where:
    - q_b: quantized bias
    - Z_x: input zero-point
    - sum(q_w): sum of quantized weights for each output channel (assumes per-channel quantization)

    Precomputing the shifted bias helps optimize inference by reducing the need to recompute
    the bias shift during each forward pass.

    Assumptions:
    - Symmetric quantization for weights.
    - Symmetric quantization for bias, with the bias scale defined as S_x * S_w.

    :param quantized_bias: [torch.IntTensor] Quantized bias (should be int32).
    :param quantized_weight: [torch.CharTensor] Quantized weights (should be int8).
    :param input_zero_point: [int] Input zero-point (int).

    :return:
        [torch.IntTensor] Shifted quantized bias tensor.
    """
    assert (quantized_bias.dtype == torch.int32), \
        f"Expected bias to be of type torch.float, but got {quantized_bias.dtype}"
    assert (isinstance(input_zero_point, int)), \
        f"Expected input zero point to be of type int, but got {input_zero_point.dtype}"

    # Sum = sum all weights over all input channel weights of each output channel
    shifted_q_bias = quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point

    return shifted_q_bias


def quantized_linear_layer(
        input_x, weight, bias, feature_bit_width, weight_bit_width, input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale):
    """
    Quantized fully-connected (linear) layer.

    The quantized output is computed as:

    q_y = (Linear[q_x, q_w] + Qbias) * (S_x * S_w / S_y) + Z_y

    Where Qbias is the shifted quantized bias:
    Qbias = q_b - Z_x * sum(q_w) (see `shift_quantized_linear_bias` for details).


    :param input_x: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bit_width: [int] quantization bit width of input and output
    :param weight_bit_width: [int] quantization bit width of weight
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale

    :return:
        [torch.CharIntTensor] quantized output feature (torch.int8)
    """
    assert input_x.dtype == torch.int8, f"Expected input dtype torch.int8, got {input_x.dtype}"
    assert weight.dtype == input_x.dtype, f"Expected weight dtype to match input, got {weight.dtype}"
    assert bias is None or bias.dtype == torch.int32, f"Expected bias dtype torch.int32, got {bias.dtype}"
    assert isinstance(input_zero_point, int), f"Expected input_zero_point to be int, got {type(input_zero_point)}"
    assert isinstance(output_zero_point, int), f"Expected output_zero_point to be int, got {type(output_zero_point)}"
    assert isinstance(input_scale, float), f"Expected input_scale to be float, got {type(input_scale)}"
    assert isinstance(output_scale, float), f"Expected output_scale to be float, got {type(output_scale)}"
    assert weight_scale.dtype == torch.float, f"Expected weight_scale to be float, got {weight_scale.dtype}"

    # Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
    if 'cpu' in input_x.device.type:
        # use 32-b MAC for simplicity
        output = torch.nn.functional.linear(input_x.to(torch.int32), weight.to(torch.int32), bias)
    else:
        # current version pytorch does not yet support integer-based linear() on GPUs
        output = torch.nn.functional.linear(input_x.float(), weight.float(), bias.float())

    # Step 2: scale the output
    # Weight_scale: [oc, 1, 1, 1] -> [oc], then expanded to [1, oc] to match output shape [batch_size, oc]
    weight_scale = weight_scale.view(1, -1)
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
        bit_width=2, batch_size=4, in_channels=8, out_channels=8):

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

    # Calculated output - precompute Q_bias
    _shifted_quantized_bias = \
        shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point)
    assert _shifted_quantized_bias.equal(shifted_quantized_bias)  # check that values match with input

    # Quantize the output
    quantized_output, output_scale, output_zero_point = linear_quantize_affine(output, bit_width)

    # Calculate the output of the fully connected layer
    _calc_quantized_output = quantized_linear_layer(
        quantized_input, quantized_weight, shifted_quantized_bias,
        bit_width, bit_width,
        input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale)
    assert _calc_quantized_output.equal(calc_quantized_output)  # check that values match with input

    reconstructed_weight = weight_scale * (quantized_weight.float() - weight_zero_point)
    reconstructed_input = input_scale * (quantized_input.float() - input_zero_point)
    reconstructed_bias = bias_scale * (quantized_bias.float() - bias_zero_point)
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
    plot_matrix(calc_quantized_output.t(), axes[2, 1], f'quantized output from quantized_linear()',
                vmin=quantized_min, vmax=quantized_max, cmap='bwr')

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


def shift_quantized_conv2d_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    Computes the shifted bias (Q_bias) for quantized convolutional layers.

    shifted_bias = q_b - Conv(Z_x, q_w)

     where:
    - q_b: quantized bias
    - Z_x: input zero-point
    - q_w: quantized weight [ch-out, ch_in, r, c]

    Note: This function does not perform an actual convolution. Instead, it sums the weights across
    the input channels and spatial dimensions, then multiplies by the input zero point of the channel
    which is a single number.

    :param quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
    :param quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
    :param input_zero_point: [int] input zero point

    :return:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert quantized_bias.dtype == torch.int32, "Expected quantized bias to be of type int32."
    assert isinstance(input_zero_point, int), "Input zero point must be an integer."

    return quantized_bias - quantized_weight.sum((1, 2, 3)).to(torch.int32) * input_zero_point


def quantized_conv2d(
        input_x, weight, bias, feature_bit_width, weight_bit_width, input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale, stride, padding, dilation, groups):
    """
    quantized 2d convolution

    q_y= (CONV[q_x,q_w] + Qbias)⋅(S_x* S_w/ S_y) + Z_y

    :param groups:
    :param dilation:
    :param padding:
    :param stride:
    :param input_x: [torch.CharTensor] quantized input (torch.int8)
    :param weight: [torch.CharTensor] quantized weight (torch.int8)
    :param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
    :param feature_bit_width: [int] quantization bit width of input and output
    :param weight_bit_width: [int] quantization bit width of weight
    :param input_zero_point: [int] input zero point
    :param output_zero_point: [int] output zero point
    :param input_scale: [float] input feature scale
    :param weight_scale: [torch.FloatTensor] weight per-channel scale
    :param output_scale: [float] output feature scale
    :return:
        [torch.(cuda.)CharTensor] quantized output feature
    """
    assert len(padding) == 4, "Padding must be a length 4 tuple."
    assert input_x.dtype == torch.int8, "Input tensor must be of type torch.int8."
    assert weight.dtype == torch.int8, "Weight tensor must be of type torch.int8."
    assert bias is None or bias.dtype == torch.int32, "Bias must be of type torch.int32 or None."
    assert isinstance(input_zero_point, int), "Input zero point must be an integer."
    assert isinstance(output_zero_point, int), "Output zero point must be an integer."
    assert isinstance(input_scale, float), "Input scale must be a float."
    assert isinstance(output_scale, float), "Output scale must be a float."
    assert weight_scale.dtype == torch.float, "Weight scale must be of type torch.float."

    # Step 1: Pad input tensor with input_zero_point
    # In quantized neural networks, the input tensor is quantized using a zero point to represent integer values.
    # Padding with the same zero point ensures that the padded areas align well with the quantization scheme,
    # making the convolution operation more consistent across the entire input tensor.
    # It is handled here separately for greater control
    input_x = torch.nn.functional.pad(input_x, padding, 'constant', value=input_zero_point)

    # Step 2: calculate integer-based 2d convolution (8-bit multiplication with 32-bit accumulation)
    if 'cpu' in input_x.device.type:
        # use 32-b MAC for simplicity
        # What should be done is 8 bit convolution, stored in 32-bit number
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

    # Step 3: Add bias if present
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)

    # Step 4: Scale the output
    weight_scale = weight_scale.view(1, -1)
    output = output * input_scale * weight_scale / output_scale

    # Step 5: Shift output by output_zero_point
    output = output + input_zero_point

    # Step 6: Clamp values to the quantized range and convert to int8
    q_min, q_max = get_quantize_range(feature_bit_width)
    output = output.round().clamp(q_min, q_max).to(torch.int8)
    return output


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


def fuse_conv_bn(conv, bn):
    """
    Fuses a BatchNorm2d layer into the preceding Conv2d layer.

    This operation merges the normalization BN layer into the weights and biases of preceding Conv2D layer,
    removing the need for a separate BatchNorm during inference.

    BatchNorm introduces floating-point operations that are inefficient on quantized hardware. By
    combining the layers, the entire convolution operation can be quantized, improving performance and
    reduces model size.

    **How It Works**:
    The fusion modifies the convolution layer’s weights and biases to include the effects of the BatchNorm
    layer's learned parameters:

    - **BatchNorm Parameters**:
        - `mu`: running mean (computed during training).
        - `sigma^2`: running variance (computed during training).
        - `gamma`: learnable scale parameter (`bn.weight`).
        - `beta`: learnable shift parameter (`bn.bias`).

    The fused weights and bias are computed as:
        - `w' = w * gamma / sqrt(sigma^2 + eps)`
        - `b' = (b - mu) * gamma / sqrt(sigma^2 + eps) + beta`

    **Benefits**:
    - Simplifies inference by merging layers.
    - Removes floating-point operations introduced by BatchNorm, making the model more efficient for
      quantization.
    - Reduces the number of layers and operations, leading to faster execution on hardware optimized
      for low-precision computations (e.g., CPUs or accelerators).

    **Reference**:
    Modified from: https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html

    :param conv: torch.nn.Conv2d: The convolutional layer to fuse.
    :param bn: torch.nn.BatchNorm2d: The BatchNorm layer to fuse.

    :return: torch.nn.Conv2d: The fused Conv2d layer, with updated weights and biases.
    """
    assert conv.bias is None, "Conv2d layer must not have a bias before fusion."

    # Compute the scaling factor (gamma / sqrt(running_var + eps))
    scale_factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)

    # Update the convolution weights: w' = w * scale_factor
    conv.weight.data = conv.weight.data * scale_factor.reshape(-1, 1, 1, 1)  # Per output channel

    # Update the convolution bias: b' = (b - mu) * scale_factor + beta
    # If no bias is present in conv, initialize it
    conv.bias = torch.nn.Parameter(- bn.running_mean.data * scale_factor + bn.bias.data)

    return conv


if __name__ == "__main__":
    plt.ion()
    random_seed = 10
    torch.random.manual_seed(random_seed)

    # # ---------------------------------------------------------------------------------------------
    # # Test Quantization Function
    # # ---------------------------------------------------------------------------------------------
    # test_linear_quantize()

    test_quantized_fc()

    # # ---------------------------------------------------------------------------------------------
    # # Quantize Model Weights
    # # ---------------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model_file = "../results_trained_models/07-06-2024_VGG_net_train_epochs_100_acc_81.pth"

    net = VGG()

    net.load_state_dict(torch.load(saved_model_file, weights_only=True))
    net.to(device)

    b_size = 128
    data_dir = './data'

    train_data_set, train_loader, test_data_set, test_loader, _ = (
        train_cifar10.get_cifar10_datasets_and_data_loaders(data_dir=data_dir, b_size=b_size))
    #
    # # Distribution of weights of the floating point model
    # plot_weight_distributions(net, extra_title='Floating point model')
    #
    # # Quantize Model and plot weight distributions
    # quantized_bit_widths = [2, 4, 8]
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
    # Fuse BM and convolutional layers to simplify quantization
    # ---------------------------------------------------------------------------------------------
    print('Before conv-bn fusion: backbone length', len(net.backbone))

    #  fuse the BN into conv layers - for less compute  quantization
    net.load_state_dict(torch.load(saved_model_file, weights_only=True))

    model_fused = copy.deepcopy(net)

    fused_backbone = []
    ptr = 0
    while ptr < len(model_fused.backbone):
        if isinstance(model_fused.backbone[ptr], torch.nn.Conv2d) and \
                isinstance(model_fused.backbone[ptr + 1], torch.nn.BatchNorm2d):
            fused_backbone.append(fuse_conv_bn(model_fused.backbone[ptr], model_fused.backbone[ptr + 1]))
            ptr += 2
        else:
            fused_backbone.append(model_fused.backbone[ptr])
            ptr += 1
    model_fused.backbone = torch.nn.Sequential(*fused_backbone)

    print('After conv-bn fusion: backbone length', len(model_fused.backbone))

    # sanity check, no BN anymore
    for m in model_fused.modules():
        assert not isinstance(m, torch.nn.BatchNorm2d)

    #  the accuracy will remain the same after fusion
    fused_acc = train_cifar10.evaluate(model_fused, test_loader, device)
    print(f'Accuracy of the fused model={fused_acc:.2f}%')

    import pdb
    pdb.set_trace()
