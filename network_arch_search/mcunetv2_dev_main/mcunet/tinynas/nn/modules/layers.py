# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from collections import OrderedDict

import torch.nn as nn
from ....utils import (
    MyModule,
    build_activation,
    get_same_padding,
    SEModule,
    ShuffleLayer,
)

__all__ = [
    "set_layer_from_config",
    "My2DLayer",
    "ConvLayer",
    "DepthConvLayer",
    "PoolingLayer",
    "IdentityLayer",
    "LinearLayer",
    "ZeroLayer",
    "MBInvertedConvLayer",
]


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        ZeroLayer.__name__: ZeroLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
    }

    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class My2DLayer(MyModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        """
        Base class for defining custom 2D layers with configurable operations order, including weight, BN,
        activation, and dropout.

        Builds a list of self modules

        Attributes:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_bn (bool): Whether to include batch normalization.
            act_func (str): Activation function to use (e.g., 'relu', 'sigmoid').
            dropout_rate (float): Dropout rate to apply after the weight operation.
            ops_order (str): Order of operations (e.g., 'weight_bn_act').

        Methods:
            forward(x): Defines the forward pass, applying all submodules in order.
            weight_op(): Abstract method for defining the weight operation; must be implemented in subclasses.
            module_str: Abstract property for generating a string representation of the module.
            config: Returns a dictionary containing the configuration of the layer.
            build_from_config(config): Abstract method to instantiate a layer from a configuration dictionary.
        """
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}

        # batch norm
        if self.use_bn:
            # bn_before_weight is a class property () see definition below. A function that can be referenced as a
            # variable. It is evaluated when called.
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm2d(in_channels)
            else:
                modules["bn"] = nn.BatchNorm2d(out_channels)
        else:
            modules["bn"] = None

        # activation
        # adds the correct activation function to dictionary of modules. If activation is First in opt_list,
        # does not do an in place operation, otherwise does in place operations.
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")

        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None

        # weight
        # THe weight_op is not defined in this base class. Instead, this is a method to force any child class to
        # define its own implementation of this function. Expects a NN module or an ordered dictionary of NN
        # layers
        modules["weight"] = self.weight_op()

        # add modules
        # add_module() is a method provided by PyTorch's nn.Module class. It dynamically adds submodules to the
        # current module instance (e.g., My2DLayer in this case) with a specified name and a corresponding
        # submodule (e.g., layers like nn.Conv2d, nn.BatchNorm2d, etc.).
        # How add_module() Works
        #     Usage: self.add_module("name", submodule)
        #     "name": A string identifier for the submodule.
        #     submodule: A PyTorch module (e.g., nn.Conv2d, nn.BatchNorm2d) to be added.
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                # dropout before weight operation
                # The dropout module is added separately before the weight operation, ensuring it is applied prior to
                # operations like convolution. However, this is a non-standard approach, as dropout is typically
                # applied after the weight operation and activation function. The reasoning behind applying dropout
                # before the weight operation remains unclear but could be a design choice or an experimental
                # regularization technique.
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                # Add all modules specified in module.weights.
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        """
        Parses ops_list. Exists on first occurrence or bn or weight. If bn comes first returns true. If
        weight comes first returns falls.
        """
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def weight_op(self):
        """ Force any derived class to implement this  function"""
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, x):
        # similar to nn.Sequential
        # n PyTorch, self._modules is an internal attribute of nn.Module (the base class for most PyTorch models).
        # It stores all the submodules of the model in a dictionary where the keys are the names of the submodules,
        # and the values are the actual submodule instances.
        #
        # When you use self._modules.values(), you're accessing all the values in this dictionary, which are the
        # submodule instances (e.g., layers, activation functions, etc.). This allows you to iterate over and
        # apply each submodule in the model sequentially
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class ConvLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        """
        A customizable 2D conv layer w/ support for group convolutions, dilation, shuffling, BN, & activation functions.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel (default: 3).
            stride (int or tuple): Stride of the convolution (default: 1).
            dilation (int): Spacing between kernel elements (default: 1).
            groups (int): Number of groups for grouped convolution (default: 1).
            bias (bool): Whether to include a bias term in the convolution (default: False).
            has_shuffle (bool): Whether to include a shuffle operation for grouped convolution (default: False).
                Shuffling in grouped convolution mixes the output channels across groups to enable better information
                exchange and improve feature diversity. It is commonly implemented by reshaping, permuting, and
                flattening the output tensor, as seen in architectures like ShuffleNet.
            use_bn (bool): Whether to include batch normalization (default: True).
            act_func (str): Activation function to use (default: "relu").
            dropout_rate (float): Dropout rate for regularization (default: 0).
            ops_order (str): Order of operations (e.g., "weight_bn_act", default: "weight_bn_act").

        Attributes:
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): Stride of the convolution.
            dilation (int): Spacing between kernel elements.
            groups (int): Number of groups for grouped convolution.
            bias (bool): Whether a bias term is used.
            has_shuffle (bool): Whether a shuffle operation is applied.

        Methods:
            weight_op():
                Constructs the weight-related operations for the layer, including convolution
                and optional shuffling for grouped convolutions.

            module_str:
                Returns a string representation of the layer configuration.

            config:
                Provides a dictionary of the current layer configuration.

            build_from_config(config):
                Reconstructs a ConvLayer instance from a configuration dictionary.

        Notes:
            - If `groups > 1` and `has_shuffle` is True, a ShuffleLayer is included after the convolution.
            - The padding is automatically calculated to maintain the spatial dimensions of the input.
        """
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        # Calls the __init__ function of the parent class.
        # Add the defined modules (BN, weight, act, dropout) into a list of sequential
        # modules and define a forward function to apply them.
        super(ConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict["conv"] = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict["shuffle"] = ShuffleLayer(self.groups)

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size

        if self.groups == 1:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedGroupConv" % (kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        return conv_str

    @property
    def config(self):
        """ Returns a dictionary containing the configuration parameters of the ConvLayer. """
        return {
            "name": ConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "has_shuffle": self.has_shuffle,

            **super(ConvLayer, self).config,
            # super(ConvLayer, self) refers to the parent class of this class (My2DLayer). it then calls the
            # config PROPERTY of the parent class.
            # ** expands the returned values into a dictionary (unpack the contents of a dictionary into
            # the current dictionary.)
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class DepthConvLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        # default normal 3x3_DepthConv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels,
            out_channels,
            use_bn,
            act_func,
            dropout_rate,
            ops_order,
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict["depth_conv"] = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.in_channels,
            bias=False,
        )
        weight_dict["point_conv"] = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            groups=self.groups,
            bias=self.bias,
        )
        if self.has_shuffle and self.groups > 1:
            weight_dict["shuffle"] = ShuffleLayer(self.groups)
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.dilation > 1:
            conv_str = "%dx%d_DilatedDepthConv" % (kernel_size[0], kernel_size[1])
        else:
            conv_str = "%dx%d_DepthConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        return conv_str

    @property
    def config(self):
        return {
            "name": DepthConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "has_shuffle": self.has_shuffle,
            **super(DepthConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)


class PoolingLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_type,
        kernel_size=2,
        stride=2,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == "avg":
            weight_dict["pool"] = nn.AvgPool2d(
                self.kernel_size,
                stride=self.stride,
                padding=padding,
                count_include_pad=False,
            )
        elif self.pool_type == "max":
            weight_dict["pool"] = nn.MaxPool2d(
                self.kernel_size, stride=self.stride, padding=padding
            )
        else:
            raise NotImplementedError
        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        return "%dx%d_%sPool" % (kernel_size[0], kernel_size[1], self.pool_type.upper())

    @property
    def config(self):
        return {
            "name": PoolingLayer.__name__,
            "pool_type": self.pool_type,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            **super(PoolingLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)


class IdentityLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(IdentityLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return "Identity"

    @property
    def config(self):
        return {
            "name": IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class LinearLayer(MyModule):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm1d(in_features)
            else:
                modules["bn"] = nn.BatchNorm1d(out_features)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # linear
        modules["weight"] = {
            "linear": nn.Linear(self.in_features, self.out_features, self.bias)
        }

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            "name": LinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


class ZeroLayer(MyModule):
    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return "Zero"

    @property
    def config(self):
        return {
            "name": ZeroLayer.__name__,
            "stride": self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)


class MBInvertedConvLayer(MyModule):
    # The SE_BASE_CHANNEL flag determines whether the SE block's reduction is based on the input channels (in_channels)
    # or the expanded channels (feature_dim). While the SE block operates after channel expansion in the inverted
    # bottleneck, using the input channels for reduction can significantly reduce the number of parameters.
    # This approach assumes that the recalibration of expanded channels can rely on shared parameters derived from
    # the original input channels, which is a reasonable trade-off between model complexity and representational
    # capacity.
    SE_BASE_CHANNEL = True

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func="relu6",
        use_se=False,
        no_dw=False,
    ):
        """
        MobileNetV2-style Inverted Residual Block with optional Squeeze-and-Excitation (SE) & depthwise conv.

        This layer implements the inverted residual block that expands the number of channels using a 1x1
        convolution, performs depthwise separable convolutions and then projects the features back down to the
        desired output channels.
        Optionally, a Squeeze-and-Excitation (SE) module can be added to recalibrate the channel-wise feature
        responses.

        :param in_channels: (int) The number of input channels.
        :param out_channels: (int) The number of output channels.
        :param kernel_size: (int, optional) The size of the convolution kernel. Default is 3.
        :param stride: (int, optional) The stride of the convolution. Default is 1.
        :param expand_ratio: (int, optional) The expansion ratio for the inverted bottleneck. Default is 6.
            The expansion ratio is the number of input channels the input is expanded to.
        :param mid_channels: (int, optional) The number of channels in the middle (bottleneck) layer. If None,
            It is calculated as `in_channels * expand_ratio`.
            Optionally instead of the expansion ratio, the actual number of mod channels can be specified.
        :param act_func: (str, optional) The activation function to use. Default is 'relu6'.
        :param use_se: (bool, optional) Whether to apply the Squeeze-and-Excitation (SE) module. Default is False.
        :param no_dw: (bool, optional) If True, depthwise convolution is skipped. Default is False.

        Attributes:
            inverted_bottleneck (nn.Module or None): A 1x1, BN and activation (expanding the input channels).
            depth_conv (nn.Module or None): A depthwise separable conv,BN  and activation.
            point_linear (nn.Module): A 1x1 conv, BN (to project the features to the output channels).

            SE_BASE_CHANNEL (bool): Flag indicating the use of base channel for SE module scaling.

            Methods:
                forward(x): Applies the inverted bottleneck, depthwise convolution, and pointwise convolution to the input tensor.
                module_str: Returns a string description of the layer configuration.
                config: Returns a dictionary of the layer's configuration parameters.
                build_from_config(config): A static method to construct the layer from a configuration dictionary.
            """
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.no_dw = no_dw

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                self.in_channels, feature_dim, 1, 1, 0, bias=False
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(feature_dim)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        pad = get_same_padding(self.kernel_size)
        depth_conv_modules = [
            (
                "conv",
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    pad,
                    groups=feature_dim,
                    bias=False,
                ),
            ),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("act", build_activation(self.act_func, inplace=True)),
        ]

        # Squeeze and Excite the activation channels
        if self.use_se:
            if self.SE_BASE_CHANNEL:
                # follow efficient to use divisor=1 for SE layer
                depth_conv_modules.append(
                    (
                        "se",
                        SEModule(
                            feature_dim, reduced_base_chs=self.in_channels, divisor=1
                        ),
                    )
                )
            else:
                depth_conv_modules.append(("se", SEModule(feature_dim)))

        if not self.no_dw:
            self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))
        else:
            self.depth_conv = None

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        if not self.no_dw:
            x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = "%dx%d_MBConv%d_%s" % (
            self.kernel_size,
            self.kernel_size,
            expand_ratio,
            self.act_func.upper(),
        )
        if self.use_se:
            layer_str = "SE_" + layer_str
        layer_str += "_O%d" % self.out_channels
        return layer_str

    @property
    def config(self):
        return {
            "name": MBInvertedConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)
