# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .my_modules import MyNetwork

__all__ = [
    "make_divisible",
    "build_activation",
    "ShuffleLayer",
    "MyGlobalAvgPool2d",
    "Hswish",
    "Hsigmoid",
    "SEModule",
    "MultiHeadCrossEntropyLoss",
]


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    elif act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class ShuffleLayer(nn.Module):
    def __init__(self, groups):
        """
        A layer that performs channel shuffling. Channel shuffling is commonly used in models like MobileNetV2 to
        enhance the efficiency of grouped convolutions by redistributing the channels across groups, promoting
        better information flow.

        Attributes:
            groups (int): The number of groups to divide the input channels into.

        Methods:
            forward(x):
                Performs the forward pass, reshaping and transposing the input tensor to shuffle the channels
                across groups.

            __repr__():
                Returns a string representation of the ShuffleLayer with the number of groups.
        """
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # reshape
        # The input tensor, of shape [batch_size, num_channels, height, width], is reshaped into
        # [batch_size, groups, channels_per_group, height, width]. The groups dimension (dimension 1) is swapped
        # with the channels dimension (dimension 2), so the tensor shape becomes
        # [batch_size, channels_per_group, groups, height, width]. Finally, the tensor is flattened back
        # into the shape [batch_size, num_channels, height, width], but now the channels have been rearranged
        # due to the transposition.
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)

        return x

    def __repr__(self):
        return "ShuffleLayer(groups=%d)" % self.groups


class MyGlobalAvgPool2d(nn.Module):
    def __init__(self, keep_dim=True):
        super(MyGlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return "MyGlobalAvgPool2d(keep_dim=%s)" % self.keep_dim


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hswish()"


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        """
        Hard Sigmoid Activation function.

        Hard sigmoid (Hsigmoid) is a computationally efficient approximation of the sigmoid activation function
        because it replaces the exponential and division operations used in the standard sigmoid with
        simple linear and piecewise operations:

        The hard sigmoid function is expressed as: HardSigmoid(x) = clip((x + 3) / 6, 0, 1).

        :param inplace: If True, the operation will be performed in-place, modifying the input tensor. Default is True.
        """
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hsigmoid()"


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, reduction=None, reduced_base_chs=None, divisor=None):
        """
        Squeeze-and-Excitation (SE) module for recalibrating channel-wise feature responses.

        :param channel: The number of input channels (features).
        :param reduction: The reduction ratio for the intermediate channel size. Defaults to 4 if not provided.
        :param reduced_base_chs: The base number of channels for reduction. If not provided, uses `channel` as the base.
        :param divisor: A divisor to ensure the intermediate channel size is divisible by this value.
        """
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction
        self.reduced_base_chs = reduced_base_chs

        # In Python, the or operator returns the first truthy value it encounters or the last value if all are falsy.
        # In the expression (reduced_base_chs or channel), it uses reduced_base_chs if it's truthy; otherwise,
        # it defaults to channel.
        num_mid = make_divisible(
            (reduced_base_chs or channel) // self.reduction,
            divisor=divisor or MyNetwork.CHANNEL_DIVISIBLE,
        )

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
                    (
                        "relu",
                        nn.ReLU6(inplace=True),
                    ),  # TODO: temporarily change to ReLU6 for easier quantization
                    ("expand", nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        """
            Global Average Pooling: Reduces the spatial dimensions by averaging each channel, capturing global context.

            1x1 Convolution (Reduce): Reduces the number of channels, creating a compact representation.
            ReLU6 Activation: Adds non-linearity while limiting outputs between 0 and 6 for quantization.
            1x1 Convolution (Expand): Expands the reduced channels back to the original number of channels.
            Hsigmoid Activation: Provides a computationally efficient approximation of sigmoid, scaling each channel’s
            importance between 0 and 1.

            Channel-wise Multiplication: The learned importance values are multiplied by the original input tensor to
            re-weight the channels.

        :param x:
        :return:
        """
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y

    def __repr__(self):
        return "SE(channel=%d, reduction=%d)" % (self.channel, self.reduction)


class MultiHeadCrossEntropyLoss(nn.Module):
    def forward(self, outputs, targets):
        assert outputs.dim() == 3, outputs
        assert targets.dim() == 2, targets

        assert outputs.size(1) == targets.size(1), (outputs, targets)
        num_heads = targets.size(1)

        loss = 0
        for k in range(num_heads):
            loss += F.cross_entropy(outputs[:, k, :], targets[:, k]) / num_heads
        return loss
