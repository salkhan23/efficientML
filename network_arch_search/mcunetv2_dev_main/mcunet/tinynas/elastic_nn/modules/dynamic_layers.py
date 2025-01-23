# Code adapted from Once for All: Train One Network and Specialize it for Efficient Deployment

from collections import OrderedDict
import copy
import torch
import torch.nn as nn

from ...nn.modules import MBInvertedConvLayer, ConvLayer, LinearLayer
from .dynamic_op import *
from ....utils import (
    adjust_bn_according_to_idx,
    copy_bn,
    make_divisible,
    SEModule,
    MyModule,
    val2list,
    get_net_device,
    build_activation,
)

__all__ = ["DynamicMBConvLayer", "DynamicConvLayer", "DynamicLinearLayer"]


class DynamicMBConvLayer(MyModule):
    SE_BASE_CHANNEL = True

    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        expand_ratio_list=6,
        stride=1,
        act_func="relu6",
        use_se=False,
        no_dw=False,
    ):
        """
        A dynamic implementation of the Mobile Inverted Bottleneck Convolution (MBConv) layer.

        This class supports runtime flexibility in input/output channels, kernel sizes,
        expansion ratios, and optional Squeeze-and-Excite (SE) modules. It is designed for
        efficient neural architecture search (NAS) and dynamic neural networks.

        :param in_channel_list: List of possible input channel sizes.
        :param out_channel_list: List of possible output channel sizes.
        :param kernel_size_list: List of possible kernel sizes for depthwise convolution. Default is [3].
        :param expand_ratio_list: List of possible expansion ratios for the inverted bottleneck. Default is [6].
        :param stride: Stride for the depthwise convolution. Default is 1.
        :param act_func: Activation function to use. Default is "relu6".
        :param use_se: Whether to include a Squeeze-and-Excite module. Default is False.
        :param no_dw: Whether to skip the depthwise convolution. Default is False.

        :raises ValueError: If invalid configurations are provided.

        method forward(x): Performs a forward pass through the dynamic MBConv layer.
        method module_str: Returns a string representation of the active configuration of the layer.
        method config: Returns the configuration dictionary of the dynamic MBConv layer.
        method build_from_config(config): Creates a DynamicMBConvLayer instance from a configuration dictionary.
        method get_active_subnet(in_channel, preserve_weight=True): Extracts a static subnet based on the current active configuration.
        method re_organize_middle_weights(expand_ratio_stage=0): Reorganizes middle layer weights based on channel importance.

        example:
            # Create a dynamic MBConv layer
            mbconv_layer = DynamicMBConvLayer(
                in_channel_list=[32, 64, 128],
                out_channel_list=[64, 128],
                kernel_size_list=[3, 5],
                expand_ratio_list=[4, 6],
                stride=2,
                use_se=True
            )

            # Forward pass with input tensor
            x = torch.randn(1, 32, 224, 224)
            y = mbconv_layer(x)

        :notes:
            - The layer dynamically adjusts its structure at runtime based on active configurations.
            - Weight reorganization and subnet extraction support efficient NAS workflows.
            - Integration with SE layers includes dynamic adjustment of reduction channels.

        :return: DynamicMBConvLayer with configurable MBConv structure.
        :rtype: DynamicMBConvLayer
        """
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = val2list(kernel_size_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        self.no_dw = no_dw

        # build modules ----------------------------------------------------------------------
        # Inverted bottleneck layer expand input channels - determine number of channels to expand to.
        max_middle_channel = round(
            max(self.in_channel_list) * max(self.expand_ratio_list)
        )

        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(

                # The inverted bottle net part uses
                # (1) a 1x1 conv to increase # of channels,
                # (2) a BN layer
                # (3) an activation function
                OrderedDict(
                    [
                        (
                            "conv", DynamicPointConv2d(
                                max(self.in_channel_list), max_middle_channel
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        if no_dw:
            self.depth_conv = None
        else:
            self.depth_conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            DynamicSeparableConv2d(
                                max_middle_channel, self.kernel_size_list, self.stride
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )
        if self.use_se:
            if self.SE_BASE_CHANNEL:
                # follow efficient to use divisor=1 for SE layer
                self.depth_conv.add_module(
                    "se",
                    DynamicSE(
                        max_middle_channel,
                        reduced_base_chs=max(self.in_channel_list),
                        divisor=1,
                    ),
                )
            else:
                self.depth_conv.add_module("se", DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicPointConv2d(
                            max_middle_channel, max(self.out_channel_list)
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = make_divisible(
                round(in_channel * self.active_expand_ratio), 8
            )

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)

        return x

    @property
    def module_str(self):
        if self.use_se:
            return "SE(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )
        else:
            return "(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )

    @property
    def config(self):
        return {
            "name": DynamicMBConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "expand_ratio_list": self.expand_ratio_list,
            "stride": self.stride,
            "act_func": self.act_func,
            "use_se": self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)

        # build the new layer
        sub_layer = MBInvertedConvLayer(
            in_channel,
            self.active_out_channel,
            self.active_kernel_size,
            self.stride,
            self.active_expand_ratio,
            act_func=self.act_func,
            mid_channels=middle_channel,
            use_se=self.use_se,
            no_dw=self.depth_conv is None
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[
                    :middle_channel, :in_channel, :, :
                ]
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        if not self.no_dw:
            sub_layer.depth_conv.conv.weight.data.copy_(
                self.depth_conv.conv.get_active_filter(
                    middle_channel, self.active_kernel_size
                ).data
            )
            copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            if self.SE_BASE_CHANNEL:
                se_mid = make_divisible(in_channel // SEModule.REDUCTION, divisor=1)
            else:
                se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid, :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.fc.reduce.bias.data[:se_mid]
            )

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel, :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.fc.expand.bias.data[:middle_channel]
            )

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[
                : self.active_out_channel, :middle_channel, :, :
            ]
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = torch.sum(
            torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3)
        )  # over input ch
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width = sorted_expand_list[expand_ratio_stage]
            target_width = round(max(self.in_channel_list) * target_width)
            importance[target_width:] = torch.arange(
                0, target_width - importance.size(0), -1
            )

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            # se expand: output dim 0 reorganize
            se_expand = self.depth_conv.se.fc.expand
            se_expand.weight.data = torch.index_select(
                se_expand.weight.data, 0, sorted_idx
            )
            se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
            # se reduce: input dim 1 reorganize
            se_reduce = self.depth_conv.se.fc.reduce
            se_reduce.weight.data = torch.index_select(
                se_reduce.weight.data, 1, sorted_idx
            )
            # middle weight reorganize
            se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
            se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

            se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
            se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
            se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)

        # TODO if inverted_bottleneck is None, the previous layer should be reorganized accordingly
        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx


class DynamicConvLayer(MyModule):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bn=True,
        act_func="relu6",
    ):
        """
        A dynamic convolutional layer that adjusts its configuration based on the provided
        input and output channel lists.

        Currently, only the maximum values from these lists are used to configure the convolutional layer.
        I am not sure how these will be used.

        :param in_channel_list:
        :param out_channel_list:
        :param kernel_size:
        :param stride:
        :param dilation:  The dilation of the convolution (default is 1).
        :param use_bn:
        :param act_func:
        """
        super(DynamicConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )

        if self.use_bn:
            # BN is done on the output of the conv operation, in the non-elastic models it can be
            # done on input channels as well.
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))

        self.act = build_activation(self.act_func, inplace=True)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x

    @property
    def module_str(self):
        return "DyConv(O%d, K%d, S%d)" % (
            self.active_out_channel,
            self.kernel_size,
            self.stride,
        )

    @property
    def config(self):
        return {
            "name": DynamicConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        """
        Creates a subnet of the current convolutional layer with a specified number of input and output channels.

        The method optionally copies the weights of the original layer into the new sub layer, including
        BN parameters if used. It is useful for tasks like dynamic pruning or layer-level adjustments where
        only a subset of the original layer's channels are needed.

        :param in_channel: (int) The number of input channels for the subnet.
        :param preserve_weight: (bool, optional) If True, the weights of the subnet are initialized by copying
            from the original layer. If False, random initialization is used. Default is True.

        :return: (nn.Module) A new convolutional layer with the specified input and output channels, optionally
             initialized with weights from the original layer.
        """
        sub_layer = ConvLayer(
            in_channel,
            self.active_out_channel,
            self.kernel_size,
            self.stride,
            self.dilation,
            use_bn=self.use_bn,
            act_func=self.act_func,
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(
            self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :]
        )

        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer


class DynamicLinearLayer(MyModule):
    def __init__(self, in_features_list, out_features, bias=True, dropout_rate=0):
        super(DynamicLinearLayer, self).__init__()

        self.in_features_list = in_features_list
        self.out_features = out_features
        self.bias = bias
        self.dropout_rate = dropout_rate

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list),
            max_out_features=self.out_features,
            bias=self.bias,
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return "DyLinear(%d)" % self.out_features

    @property
    def config(self):
        return {
            "name": DynamicLinear.__name__,
            "in_features_list": self.in_features_list,
            "out_features": self.out_features,
            "bias": self.bias,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        sub_layer = LinearLayer(
            in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate
        )
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_(
            self.linear.linear.weight.data[: self.out_features, :in_features]
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(
                self.linear.linear.bias.data[: self.out_features]
            )
        return sub_layer
