import os
import torch
import torch.nn as nn
import torch.distributed as dist


def dist_init(world_size, rank):
    """
    Init the pytorch distributed communication lib
    Args:
        rank: (int) the device ID
        world_size: (int) the total devices
    """
    # change it to the corresponding ip addr
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    assert dist.is_initialized(), "Error! The distributed env is not initialized!"

    return True


def cleanup():
    """
    destroy the communication group
    """
    dist.destroy_process_group()


def get_local_rank():
    """
    get the local rank (devices id)
    """
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def init_parameters(model):
    """
    Boradcast the initial gradients of the model parameters、
    """
    if get_world_size() > 1:
        # implement your own init
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def allreduce_average_gradients(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        # implement your own aggregation method
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_gradients_and_bn(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            dist.all_reduce(mod.running_mean.data, op=dist.ReduceOp.SUM)
            dist.all_reduce(mod.running_var.data, op=dist.ReduceOp.SUM)
            mod.running_mean.data /= size
            mod.running_var.data /= size
