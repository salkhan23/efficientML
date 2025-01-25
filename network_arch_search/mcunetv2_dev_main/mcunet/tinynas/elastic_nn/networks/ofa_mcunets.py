# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

from ..modules import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
from ...nn.modules import ConvLayer, IdentityLayer, LinearLayer, MBInvertedConvLayer
from ...nn.networks import MCUNets, MobileInvertedResidualBlock
from ....utils import make_divisible, val2list

__all__ = ["OFAMCUNets"]


class OFAMCUNets(MCUNets):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.1,
        base_stage_width=None,
        base_depth=(2, 2, 2, 2, 2),
        width_mult_list=1.0,
        ks_list=3,
        expand_ratio_list=6,
        depth_list=4,
        no_mix_layer=False,
        fuse_blk1=False,
        se_stages=None,
    ):
        """
        Once-For-All MCUNets (OFAMCUNets): Extends MCUNets to support dynamic configurations for:
            - Width (channel scaling)
            - Kernel size
            - Expansion ratio
            - Depth

        This design is inspired by the "Once-for-All" network approach, allowing flexible adjustments
        to optimize performance across a range of hardware constraints and deployment scenarios.

        Reference:
            - Paper: https://arxiv.org/abs/1908.09791

        Parameters
        ----------
        n_classes : int, optional
            Number of output classes for classification (default=1000).
        bn_param : tuple, optional
            Batch normalization parameters as (momentum, epsilon) (default=(0.1, 1e-3)).
        dropout_rate : float, optional
            Dropout rate for the final classifier layer (default=0.1).
        base_stage_width : list, str, or None, optional
            Base channel widths for each stage. Can be:
            - Predefined configurations: ["google", "proxyless", "mcunet384"]
            - Custom list of widths.
        base_depth : list or tuple, optional
            Base number of inverted residual blocks per stage (default=(2, 2, 2, 2, 2)).
        width_mult_list : float, list, or np.ndarray, optional
            Scaling factors for channel widths (default=1.0).
        ks_list : int, list, or np.ndarray, optional
            Supported kernel sizes for dynamic layers (default=3).
        expand_ratio_list : int, list, or np.ndarray, optional
            Expansion ratios for inverted residual layers (default=6).
        depth_list : int, list, or np.ndarray, optional
            Depth configurations for intermediate stages (default=4).
        no_mix_layer : bool, optional
            If True, disables the feature mix layer to reduce model size (default=False).
        fuse_blk1 : bool, optional
            If True, removes depthwise convolution in the first block (default=False).
        se_stages : list of bool, optional
            Specifies whether Squeeze-and-Excitation (SE) blocks are applied in each stage.

        Attributes
        ----------
        width_mult_list : list
            Sorted list of width multipliers.
        ks_list : list
            Sorted list of kernel sizes.
        expand_ratio_list : list
            Sorted list of expansion ratios.
        depth_list : list
            Sorted list of depths.
        runtime_depth : list
            Active depth for each block group during runtime.

        Methods
        -------
        name() -> str
            Returns the name of the model.
        forward(x)
            Performs the forward pass through the network.

        Examples
        --------
        # >>> model = OFAMCUNets(n_classes=10, base_stage_width="proxyless")
        # >>> output = model(torch.randn(1, 3, 224, 224))
        # >>> print(output.shape)  # (1, 10)
        """
        self.width_mult_list = val2list(width_mult_list, 1)
        self.ks_list = val2list(ks_list, 1)  # kernel size list.
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.base_stage_width = base_stage_width
        self.base_depth = base_depth

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        # Base number of output channels in the network.
        # Specifying a string such as  'google' loads a predefined set.
        if base_stage_width == "google":  # mobileNetV2
            base_stage_width = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        elif base_stage_width == "proxyless":  # proxyless NAS builds on mobileNet V2.
            # ProxylessNAS Stage Width
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 1280]
        elif base_stage_width == "mcunet384":
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 384]

        # (1) Creates a list of # of input channels for each width multiplier in self.width_mult_list.
        # (2) Divide by 8: Using multiples of 8 in SIMD (Single instruction multiple data) HW operations aligns with
        # the HW arch. of GPUs and AI accelerators. This alignment ensures better utilization of parallel processing
        # units, leading to faster computation and resource efficiency.
        input_channel = [
            make_divisible(base_stage_width[0] * width_mult, 8)
            for width_mult in self.width_mult_list
        ]
        # E.g. if width_mult_list=[0.5, 0.75, 1.0], input_channel = [16, 24, 32]

        first_block_width = [
            make_divisible(base_stage_width[1] * width_mult, 8)
            for width_mult in self.width_mult_list
        ]
        # E.g. if width_mult_list = [0.5, 0.75, 1.0], first_block_width = [8, 16, 16]

        last_channel = [
            make_divisible(base_stage_width[-1] * width_mult, 8)
            if width_mult > 1.0
            else base_stage_width[-1]
            for width_mult in self.width_mult_list
        ]
        # E.g. if width_mult_list = [0.5, 0.75, 1.0], last channel = [384, 384, 384]

        # first conv layer -----------------------------------------------------------------------
        # fixed number of input channels. Simple Conv Layer (con, BN, act)
        if len(input_channel) == 1:
            first_conv = ConvLayer(
                3,
                max(input_channel),
                kernel_size=3,
                stride=2,
                use_bn=True,
                act_func="relu6",
                ops_order="weight_bn_act",
            )
        else:
            first_conv = DynamicConvLayer(
                in_channel_list=val2list(3, len(input_channel)),  # if input_channels 3, = [3, 3, 3]
                out_channel_list=input_channel,
                kernel_size=3,
                stride=2,
                act_func="relu6",
            )
        # first block -----------------------------------------------------------------------------
        # Single inverted block (
        #   (1x1 conv increase ch, BN, act)
        #   (depthwise 3x3, BN, act, squeeze_n_excite)
        #   (1x1 conv decrease chan, BN)
        # )
        if len(first_block_width) == 1:
            first_block_conv = MBInvertedConvLayer(
                in_channels=max(input_channel),
                out_channels=max(first_block_width),
                kernel_size=3,
                stride=1,
                expand_ratio=1,
                act_func="relu6",
                no_dw=fuse_blk1,
            )
        else:
            first_block_conv = DynamicMBConvLayer(
                in_channel_list=input_channel,
                out_channel_list=first_block_width,
                kernel_size_list=3,
                expand_ratio_list=1,
                stride=1,
                act_func="relu6",
                no_dw=fuse_blk1,
            )

        # No residual connection on first block
        first_block = MobileInvertedResidualBlock(first_block_conv, None)

        input_channel = first_block_width

        # inverted residual blocks  -----------------------------------------------------
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1

        stride_stages = [2, 2, 2, 1, 2, 1]  # conv stride used in each stage. Net has 6 intermediate stages

        # The depth of each intermediate stage
        # (number of inverted bottleneck blocks  (n_blockd) in each stage)
        if depth_list is None:
            n_block_list = [2, 3, 4, 3, 3, 1]
            self.depth_list = [4, 4]
            print("Use MobileNetV2 Depth Setting")
        else:
            n_block_list = [
                max(self.depth_list) + self.base_depth[i] for i in range(5)
            ] + [1]  # This appends the number 1 as the last element of n_block_list.

        # The widths of each block
        width_list = []
        for base_width in base_stage_width[2:-1]:
            #  base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 384]
            #  self.width_mult_list = [0.5, 0.75, 1.0]
            width = [
                make_divisible(base_width * width_mult, 8)
                for width_mult in self.width_mult_list
            ]
            width_list.append(width)
        # [[16, 16, 24], [16, 32, 40], [40, 56, 80], [48, 72, 96], [96, 144, 192], [160, 240, 320]]

        if se_stages is None:
            se_stages = [False] * (len(base_stage_width) - 3)
            assert len(se_stages) == len(width_list)

        # for each stage
        # Construct the intermediate stages of the network:
        # For each stage, defined by its width, number of blocks, stride, and SE (Squeeze-and-Excitation) usage:
        # 1. Append the block indices for the current stage to `self.block_group_info`.
        # 2. For each block in the stage:
        #    - Set the stride for the first block in the stage (`s`), and use stride 1 for subsequent blocks.
        #    - Create a dynamic MobileNetV2-style inverted residual block (`DynamicMBConvLayer`):
        #    - Add a shortcut connection (identity layer) if the stride is 1 and input/output channels match.
        #    - Append the constructed block to the `blocks` list.
        # 3. Update the input channels for the next block or stage.
        for width, n_block, s, use_se in zip(
            width_list, n_block_list, stride_stages, se_stages
        ):
            # self.block_group_info is a list of lists that keeps track of the block indices for each
            # stage in the network. Each sublist in self.block_group_info corresponds to a stage in the network and
            # contains the indices of the blocks that belong to that stage.
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1

                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(input_channel, 1),
                    out_channel_list=val2list(output_channel, 1),
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func="relu6",
                    use_se=use_se[i] if isinstance(use_se, list) else use_se,
                )

                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = MobileInvertedResidualBlock(
                    mobile_inverted_conv, shortcut
                )

                blocks.append(mb_inverted_block)
                input_channel = output_channel

        # 1x1_conv before global average pooling
        if no_mix_layer:  # remove mix layer to reduce model size
            feature_mix_layer = None
            if len(self.width_mult_list) == 1:
                classifier = LinearLayer(
                    max(input_channel), n_classes, dropout_rate=dropout_rate
                )
            else:
                classifier = DynamicLinearLayer(
                    in_features_list=input_channel,
                    out_features=n_classes,
                    bias=True,
                    dropout_rate=dropout_rate,
                )
        else:
            if len(last_channel) == 1:
                feature_mix_layer = ConvLayer(
                    max(input_channel),
                    max(last_channel),
                    kernel_size=1,
                    use_bn=True,
                    act_func="relu6",
                )
                classifier = LinearLayer(
                    max(last_channel), n_classes, dropout_rate=dropout_rate
                )
            else:
                feature_mix_layer = DynamicConvLayer(
                    in_channel_list=input_channel,
                    out_channel_list=last_channel,
                    kernel_size=1,
                    stride=1,
                    act_func="relu6",
                )
                classifier = DynamicLinearLayer(
                    in_features_list=last_channel,
                    out_features=n_classes,
                    bias=True,
                    dropout_rate=dropout_rate,
                )

        # Initialize the parent class (MCUNets) with the network components:
        # - first_conv: The initial convolutional layer for input processing.
        # - blocks: A list of MobileInvertedResidualBlock layers for intermediate feature extraction.
        # - feature_mix_layer: An optional 1x1 convolutional layer for feature mixing before classification.
        # - classifier: The final fully connected layer for output classification.
        super(OFAMCUNets, self).__init__(
            first_conv, blocks, feature_mix_layer, classifier
        )

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return "OFAMCUNets"

    def forward(self, x):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        # feature_mix_layer
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)

        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        _str += self.blocks[0].module_str + "\n"

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"
        if self.feature_mix_layer is not None:
            _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": OFAMCUNets.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "feature_mix_layer": None
            if self.feature_mix_layer is None
            else self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError("do not support this function")

    def load_weights_from_net(self, proxyless_model_dict):
        model_dict = self.state_dict()
        for key in proxyless_model_dict:
            if key in model_dict:
                new_key = key
            elif ".bn.bn." in key:
                new_key = key.replace(".bn.bn.", ".bn.")
            elif ".conv.conv.weight" in key:
                new_key = key.replace(".conv.conv.weight", ".conv.weight")
            elif ".linear.linear." in key:
                new_key = key.replace(".linear.linear.", ".linear.")
            ##############################################################################
            elif ".linear." in key:
                new_key = key.replace(".linear.", ".linear.linear.")
            elif "bn." in key:
                new_key = key.replace("bn.", "bn.bn.")
            elif "conv.weight" in key:
                new_key = key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = proxyless_model_dict[key]
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, wid=None, ks=None, e=None, d=None, **kwargs):

        # 1. Set the active width multiplier (`wid`) to scale the number of output channels for all layers.
        #    - If `wid` is provided, use the corresponding value from `out_channel_list`.
        #    - If `wid` is None, use the maximum value from `out_channel_list`.
        #    - Note: Adjusting output channels here requires the input channels of the next layer to be updated.
        #      This is handled dynamically in the forward pass of each layer.
        #    -  In the forward pass:
        #    - The inverted bottleneck layer dynamically adjusts its output channels based on the input channels
        #      and expansion ratio.
        #    - The depthwise convolution adjusts its kernel size.
        #    - The point wise convolution adjusts its output channels to match `.active_out_channel`.

        # width_mult_id = val2list(wid, 3 + len(self.block_group_info))
        # print(' * Using a wid of ', wid)
        for m in self.modules():
            if hasattr(m, "out_channel_list"):
                if wid is not None:
                    m.active_out_channel = m.out_channel_list[wid]
                else:
                    m.active_out_channel = max(m.out_channel_list)

        # n_channel_choices = [len(m.out_channel_list) for m in self.modules() if hasattr(m, 'out_channel_list')]
        # print(n_channel_choices)
        # exit()
        # def set_output_channel(m):
        #     if hasattr(m, 'active_out_channel') and hasattr(m, 'out_channel_list'):
        #         m.active_out_channel = make_divisible(max(m.out_channel_list) * wid, 8)
        # set_output_channel(self.first_conv)
        # set_output_channel(self.feature_mix_layer)
        # for b in self.blocks:
        #     set_output_channel(b.mobile_inverted_conv)

        # 2. Adjust kernel size and expansion ration
        # ks: Kernel sizes for the convolutional layers in each block.
        # e: Expansion ratios for the inverted residual blocks.
        # Why -1? Because the first block in the network is treated differently:
        # The first block is typically a fixed block (e.g., the first conv layer or the first inverted residual block).
        # It does not participate in the dynamic configuration of kernel sizes or expansion ratios.
        # ks and expand_ratio are applied only to the remaining blocks (i.e., self.blocks[1:]).
        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)

        depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.mobile_inverted_conv.active_kernel_size = k
            if e is not None:
                block.mobile_inverted_conv.active_expand_ratio = e

        # 3. Adjust the depth (number of blocks) for each stage.
        # Depth is treated differently. It is per stage vs per block

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d) + (
                    self.base_depth[i] if i < len(self.base_depth) else 1
                )

    def set_constraint(self, include_list, constraint_type="depth"):
        # only used for progressive shrinking
        if constraint_type == "depth":
            self.__dict__["_depth_include_list"] = include_list.copy()
        elif constraint_type == "expand_ratio":
            self.__dict__["_expand_include_list"] = include_list.copy()
        elif constraint_type == "kernel_size":
            self.__dict__["_ks_include_list"] = include_list.copy()
        elif constraint_type == "width_mult":
            self.__dict__["_widthMult_include_list"] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__["_depth_include_list"] = None
        self.__dict__["_expand_include_list"] = None
        self.__dict__["_ks_include_list"] = None
        self.__dict__["_widthMult_include_list"] = None

    def sample_active_subnet(self, sample_function=random.choice, image_size=None):
        """
        Sample from the  possible network settings and create a new config. Set the active config of the model to
        this new configuration
        :param sample_function:
        :param image_size:
        :return:
        """
        # Allowed kernel sizes (self.ks_list)
        ks_candidates = (
            self.ks_list
            if self.__dict__.get("_ks_include_list", None) is None
            else self.__dict__["_ks_include_list"]
        )

        # Expansion ratios for inverted bottleneck layers
        expand_candidates = (
            self.expand_ratio_list
            if self.__dict__.get("_expand_include_list", None) is None
            else self.__dict__["_expand_include_list"]
        )

        # Depth ratio for different stages of the network.
        depth_candidates = (
            self.depth_list
            if self.__dict__.get("_depth_include_list", None) is None
            else self.__dict__["_depth_include_list"]
        )

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
            # Creates a list of lists for kernel size candidates for each block.
            # self.blocks are defined in the parent block. They are initialized with the super call.
            # They are the intermediate layers (first blocks, stages2, 3, 4, 5)

        for k_set in ks_candidates:
            k = sample_function(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = sample_function(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [
                depth_candidates for _ in range(len(self.block_group_info))
            ]
        for d_set in depth_candidates:
            d = sample_function(d_set)
            depth_setting.append(d)

        # sample width_mult, move to last to keep the same randomness
        width_mult_setting = sample_function(range(0, len(self.width_mult_list) - 1))

        self.set_active_subnet(
            width_mult_setting, ks_setting, expand_setting, depth_setting
        )

        cfg = {
            "wid": width_mult_setting,
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }
        if image_size is not None:
            cfg.update(image_size=image_size)

        return cfg

    def get_active_subnet(self, preserve_weight=True):
        def get_or_copy_subnet(m, **kwargs):
            if hasattr(m, "get_active_subnet"):
                out = m.get_active_subnet(preserve_weight=preserve_weight, **kwargs)
            else:
                out = copy.deepcopy(m)
            return out

        first_conv = get_or_copy_subnet(self.first_conv, in_channel=3)
        input_channel = first_conv.out_channels

        blocks = [
            MobileInvertedResidualBlock(
                get_or_copy_subnet(
                    self.blocks[0].mobile_inverted_conv, in_channel=input_channel
                ),
                copy.deepcopy(self.blocks[0].shortcut),
            )
        ]

        input_channel = blocks[0].mobile_inverted_conv.out_channels

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    MobileInvertedResidualBlock(
                        self.blocks[idx].mobile_inverted_conv.get_active_subnet(
                            input_channel, preserve_weight
                        ),
                        copy.deepcopy(self.blocks[idx].shortcut),
                    )
                )
                input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
            blocks += stage_blocks

        feature_mix_layer = get_or_copy_subnet(
            self.feature_mix_layer, in_channel=input_channel
        )
        input_channel = (
            feature_mix_layer.out_channels
            if feature_mix_layer is not None
            else input_channel
        )
        classifier = get_or_copy_subnet(self.classifier, in_features=input_channel)

        _subnet = MCUNets(first_conv, blocks, feature_mix_layer, classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        if len(self.width_mult_list) > 1:
            print(
                " * WARNING: sorting is not implemented right for multiple width-mult"
            )

        for block in self.blocks[1:]:
            block.mobile_inverted_conv.re_organize_middle_weights(expand_ratio_stage)
