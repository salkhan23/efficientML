# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch

from ....utils import get_same_padding, sub_filter_start_end, make_divisible, SEModule

__all__ = [
    "DynamicSeparableConv2d",
    "DynamicPointConv2d",
    "DynamicLinear",
    "DynamicBatchNorm2d",
    "DynamicSE",
]


class DynamicSeparableConv2d(nn.Module):
    # Haotian: official version uses KERNEL_TRANSFORM_MODE=None,
    # but the ckpt requires it to be 1
    KERNEL_TRANSFORM_MODE = 1

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small**2)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :out_channel, :in_channel, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks**2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, in_channel)
        return y


class DynamicPointConv2d(nn.Module):
    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        """
        A dynamic 2D pointwise convolutional layer that adjusts the number of input and output channels
        based on the provided configuration. This layer allows for flexible adjustment of the active
        output channels during the forward pass.

        Pointwise convolutions use a kernel size of 1x1, which allows for transforming the input across
        its channels without affecting the spatial dimensions. This operation helps to adjust the number
        of channels while maintaining the resolution of the input feature map.

        @param max_in_channels: The maximum number of input channels that this convolutional layer can handle.
        @param max_out_channels: The maximum number of output channels that this convolutional layer can produce.
        @param kernel_size: The size of the convolutional kernel (default is 1).
        @param stride: The stride of the convolution (default is 1).
        @param dilation: The dilation of the convolution (default is 1).

        @attr max_in_channels: The maximum number of input channels.
        @attr max_out_channels: The maximum number of output channels.
        @attr kernel_size: The size of the convolutional kernel.
        @attr stride: The stride of the convolution.
        @attr dilation: The dilation of the convolution.
        @attr conv: A standard PyTorch 2D convolutional layer that uses the maximum input and output channels.
        @attr active_out_channel: The number of active output channels used in the forward pass, initially set to the maximum output channels.

        Methods:
        --------
        forward(x, out_channel=None):
            Forward pass through the dynamic convolutional layer. If `out_channel` is not provided, it uses the
            `active_out_channel`. The method performs a convolution using the specified number of output channels
            and applies the same padding and dilation as the original configuration.
        """
        super(DynamicPointConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def forward(self, x, out_channel=None):
        """
        Different from regular forward function of a conv layer, a # of out_channels can be specified
        for dynamic size adjustments.
        """
        if out_channel is None:
            out_channel = self.active_out_channel

        in_channel = x.size(1)

        # Select a subset of the convolution weights based on the specified input and output channels.
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()

        padding = get_same_padding(self.kernel_size)

        # Perform the convolution using F.conv2d, as the weight tensor size has been modified.
        # Alternatively, the convolution could be done directly using the layer's .conv attribute,
        # but this would require reshaping the weights back to their full size, which is less efficient.
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)

        return y


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.linear.weight[:out_features, :in_features].contiguous()
        bias = self.linear.bias[:out_features] if self.bias else None
        y = F.linear(x, weight, bias)
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        """
        A dynamic implementation of 2D batch normalization that supports flexible feature dimensions.
        The batch normalization operation adapts to the feature dimension specified at runtime,
        enabling efficient use of BN even for dynamically varying input feature sizes.

        :param max_feature_dim:
        """
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        """
        The method bn_forward handles BN when the input tensor has a different # of channels than the
        BatchNorm2d layer was trained with. This adjustment is necessary because batch normalization relies
        on running statistics (mean and variance) that are calculated per feature/channel during training.
        When the input tensor has more channels than expected, the BatchNorm2d layer cannot directly apply the
        existing statistics for the additional channels.

        Thus, the method adjusts the running statistics dynamically. It selects a subset of the statistics
        corresponding to the input channels (feature_dim), and adjusts how the statistics are updated—using
        either cumulative averaging (over all batches) or exponential moving average. This ensures that
        normalization works even when the input tensor's feature dimension differs from the one the model
        was originally trained on, maintaining correct normalization behavior and preventing errors.

        :param x: (Tensor) The input tensor to be normalized, typically with shape [b_size, n_features, h, w].
        :param bn: (nn.BatchNorm2d): The BatchNorm2d layer containing the normalization
               parameters such as running mean, variance, weights, and biases.
        :param feature_dim: (int): The feature dimension (number of channels) of the input tensor `x`.
               This value is used to adjust the batch normalization parameters when
               the input's channel size differs from the initialized size in the layer.

        :return:
        """
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            # SET_RUNNING_STATISTICS is a flag in DynamicBatchNorm2d that controls whether running statistics
            # (mean and variance) are used during batch normalization. Setting it to False means the
            # running statistics are only used if the input feature dimensions match the expected ones;
            # otherwise, batch normalization will compute statistics from the current batch.
            return bn(x)

        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1

                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum

            # This function uses the functional F.batch_norm to perform batch normalization instead of the layer's
            # built-in nn.BatchNorm2d. This approach allows for performing BN on fewer channels than the total
            # number of channels, by selecting only the channels up to the specified number of feature maps.
            # This helps handle scenarios where the input tensor has fewer channels than originally trained for or
            # when adjustments to the feature dimensions are needed.
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicSE(SEModule):
    def __init__(
        self, max_channel, reduction=None, reduced_base_chs=None, divisor=None
    ):
        super(DynamicSE, self).__init__(
            max_channel, reduction, reduced_base_chs, divisor
        )

    def forward(self, x):
        in_channel = x.size(1)
        if self.reduced_base_chs is None:
            num_mid = make_divisible(in_channel // self.reduction, divisor=8)
        else:
            num_mid = make_divisible(self.reduced_base_chs // self.reduction, divisor=1)

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid, :in_channel, :, :].contiguous()
        reduce_bias = (
            reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        )
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel, :num_mid, :, :].contiguous()
        expand_bias = (
            expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        )
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y
