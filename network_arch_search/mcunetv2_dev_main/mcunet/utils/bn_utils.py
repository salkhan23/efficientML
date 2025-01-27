# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy

import torch.nn.functional as F
import torch.nn as nn
import torch

__all__ = ['set_running_statistics', 'adjust_bn_according_to_idx', 'copy_bn']


def set_running_statistics(model, data_loader, distributed=False, maximum_iter=-1):
    from .common_tools import AverageMeter
    from .pytorch_utils import get_net_device
    from ..tinynas.elastic_nn.modules import DynamicBatchNorm2d

    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)

    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if distributed:
                raise NotImplementedError
            else:
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                """ mean_est and var_est are AverageMeter objects used to track the running mean and variance
                over all batches during the iteration."""
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x, batch_mean, batch_var, bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim], False,
                        0.0, bn.eps,
                    )

                return lambda_forward

            # Replace the BatchNorm layer's forward pass with a custom implementation. This custom forward function
            # utilizes a Python closure. A closure is a function that retains access to variables from its enclosing
            # scope, even after the outer function has finished executing.
            # The `new_forward` function is initialized with the `mean_est` and `var_est` AverageMeter objects, which
            # track the running mean and variance for each batch. This function returns `lambda_forward`, which
            # maintains the same interface as the original forward function but has access to the `mean_est` and
            # `var_est` for that particular layer, ensuring correct tracking of statistics during training.
            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    with torch.no_grad():

        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        # Set the flag to enable the use of running statistics (mean and variance) during batch normalization

        for i_iter, (images, _) in enumerate(data_loader):
            if i_iter == maximum_iter:
                break

            # If the input image has a single channel, repeat it across 3 channels
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)

            images = images.to(get_net_device(forward_model))

            forward_model(images)

        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            # Get the number of channels (first dimension) from the stored mean
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)

            # store the new mean and variance in the original model.
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

    del forward_model


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
    bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    """
    Each BN layer has multiple parameters. (weights and biases of each channel and also the running mean and
    variances). Need to copy them over manually

    :param target_bn:
    :param src_bn:

    :return:
    """
    feature_dim = target_bn.num_features

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
    target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])
