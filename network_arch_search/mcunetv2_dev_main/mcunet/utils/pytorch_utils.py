import copy
import torch
import torch.nn as nn

__all__ = [
    "rm_bn_from_net",
    "get_net_device",
    "count_parameters",
    "count_net_flops",
    "count_peak_activation_size",
]

""" Network profiling """


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x  # exclude from computation
            del m.weight  # exclude model size
            del m.bias
            del m.running_mean
            del m.running_var


def rm_bn(module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module_output = nn.Identity()

    for name, child in module.named_children():
        module_output.add_module(name, rm_bn(child))
    del module
    return module_output


def get_net_device(net):
    """
    Retrieves the device of the first parameter in the model (usually the weights of the first layer).
    Since all model parameters are typically placed on the same device, checking the device of the first parameter
    provides a quick and reliable way to determine the device the model is on. The next() function extracts the
    first parameter from the iterator returned by net.parameters().

    net.parameters() returns an iterator, not a list or array. In Python, an iterator doesn't allow direct
    indexing like parameters[0]. Using __next__() retrieves the first item from the iterator, which in this
    case is the first parameter of the model.

    :param net:
    :return:
    """
    return net.parameters().__next__().device


def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


def count_net_flops(model, data_shape):
    from thop import profile
    # The thop module is developed and maintained by The Algorithmic Engineering Team at Mobile AI.
    # It is primarily designed for estimating the number of floating-point operations (FLOPs) and the
    # number of parameters in PyTorch models https://github.com/Lyken17/pytorch-OpCounter

    model = copy.deepcopy(model)
    rm_bn_from_net(model)  # remove bn since it is eventually fused (by the AI compiler)

    total_macs, _ = profile(
        model,
        inputs=(torch.randn(*data_shape).to(get_net_device(model)),),
        verbose=False,
    )
    del model
    return total_macs


def count_peak_activation_size(net, data_shape=(1, 3, 224, 224)):
    """
    Calculates the PEAK memory size of ACTIVATIONS during the forward pass of a neural network.
    (max input/output activations)

    This function analyzes the given network and computes the maximum memory required for storing
    the activations (input and output sizes) of the layers during the forward pass. It is useful
    for determining the memory consumption of a network, particularly in the context of resource-constrained
    environments like mobile devices or embedded systems.

    Args:
    net (nn.Module): The PyTorch model whose peak activation memory usage is to be calculated.
    data_shape (tuple, optional): The shape of the input data, default is (1, 3, 224, 224) for a single image.
        It should be a tuple representing the batch size and input dimensions.

    Returns:
        float: The peak memory size of activations in the network, in number of elements (floats).

    Explanation:
        The function works by:
        1. Registering hooks to track the input and output sizes of each module.
        2. Running a forward pass with a random input tensor to collect the activation sizes.
        3. Summing up the activation sizes of the relevant layers and blocks (convolutional and linear layers).
        4. Returning the maximum memory size encountered during the pass.
    """
    from ..tinynas.nn.networks import MobileInvertedResidualBlock

    def record_in_out_size(m, x, y):
        x = x[0]  # x[0] extracts the input tensor for just the first sample in the batch,
        # This is typically used when calculating the number of elements (via numel()) for the input size.

        m.input_size = torch.Tensor([x.numel()])
        m.output_size = torch.Tensor([y.numel()])

    def add_io_hooks(m_):
        m_type = type(m_)
        if m_type in [nn.Conv2d, nn.Linear, MobileInvertedResidualBlock]:
            #  register a buffer ( non-trainable tensors) named "input_size" to the module m_ & initializing it
            #  with a tensor of zeros (torch.zeros(1)). This buffer is later updated during the forward pass
            #  with the input size for that module (see record_in_out_size).
            m_.register_buffer("input_size", torch.zeros(1))

            m_.register_buffer("output_size", torch.zeros(1))
            m_.register_forward_hook(record_in_out_size)

    def count_conv_mem(m):
        # we assume we only need to store input and output, the weights are partially loaded for computation
        if m is None:
            return 0
        if hasattr(m, "conv"):
            m = m.conv
        elif hasattr(m, "linear"):
            m = m.linear
        assert isinstance(m, (nn.Conv2d, nn.Linear))
        return m.input_size.item() + m.output_size.item()

    def count_block(m):
        from ..tinynas.nn.modules import ZeroLayer

        assert isinstance(m, MobileInvertedResidualBlock)

        if m.mobile_inverted_conv is None or isinstance(
            m.mobile_inverted_conv, ZeroLayer
        ):  # just an identical mapping
            return 0
        elif m.shortcut is None or isinstance(
            m.shortcut, ZeroLayer
        ):  # no residual connection, just convs
            return max(
                [
                    count_conv_mem(m.mobile_inverted_conv.inverted_bottleneck),
                    count_conv_mem(m.mobile_inverted_conv.depth_conv),
                    count_conv_mem(m.mobile_inverted_conv.point_linear),
                ]
            )
        else:  # convs and residual
            residual_size = (
                m.mobile_inverted_conv.inverted_bottleneck.conv.input_size.item()
            )
            # consider residual size for later layers
            return max(
                [
                    count_conv_mem(m.mobile_inverted_conv.inverted_bottleneck),
                    count_conv_mem(m.mobile_inverted_conv.depth_conv) + residual_size,
                    # TODO: can we omit the residual here? reuse the output?
                    count_conv_mem(
                        m.mobile_inverted_conv.point_linear
                    ),  # + residual_size,
                ]
            )

    # Check if the input net is an instance of nn.DataParallel. nn.DataParallel is a PyTorch module wrapper
    # used to parallelize computations across multiple GPUs. If the model is wrapped in nn.DataParallel,
    # the actual model is stored in the module attribute of the nn.DataParallel object.
    if isinstance(net, nn.DataParallel):
        net = net.module
    net = copy.deepcopy(net)  # done to modify the model, with hooks etc.

    from ..tinynas.nn.networks import MCUNets

    assert isinstance(net, MCUNets)

    # record the input and output size
    # 1. net.apply(add_io_hooks) traverses the entire module hierarchy of net, including all submodules
    #    and nested layers, and applies the function add_io_hooks to each module.
    #    How it operates:
    #    - Starts with the top-level module (the main net model itself).
    #    - Recursively descends into all child modules and submodules.
    #    - Calls the provided function (add_io_hooks) on each module.
    # 2. The apply_io_hooks function expects a single parameter (the module being processed),
    #    and net.apply() handles this by automatically passing each module as an argument when it
    #    calls the function.
    net.apply(add_io_hooks)

    # Pass a random input to the model to get its input and ouptut sizes via the hooks.
    with torch.no_grad():
        _ = net(torch.randn(*data_shape).to(net.parameters().__next__().device))

    mem_list = [
        count_conv_mem(net.first_conv),
        count_conv_mem(net.feature_mix_layer),
        count_conv_mem(net.classifier),
    ] + [count_block(blk) for blk in net.blocks]

    del net
    return max(mem_list)  # pick the peak mem
