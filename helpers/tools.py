import torch
from torch.optim import SGD, Adam
from torchvision.models import resnet18, resnet50
import numpy as np
import random
import wandb
import timm
import gpustat
import os

##### custom files
from models import *
from dp_optimizers import DPMicroAdam_v1
from dp_optimizers import AdamBC
from dp_optimizers import MicroAdam
from dp_optimizers import DPMicroAdamStP
from dp_optimizers import DPAdamStP
from dp_optimizers import DPSGD

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def setup_wandb(project, job_type, group, name, config):
    return wandb.init(
        project=project,
        job_type=job_type,
        entity='',
        group=group,
        name=name,
        config=config,
        settings=wandb.Settings(start_method='fork'))


def get_model(args): # model_name, dataset_name, args, pretrained=False):
    model_name = args.model_name
    dataset_name = args.dataset_name
    pretrained = args.model_pretrained

    num_classes = dict(mnist=10,
                       rotmnist=10,
                       permmnist=10,
                       imagenette=10,
                       imagewoof=10,
                       cifar10=10,
                       cifar10_fine=10,
                       cifar10_no_aug=10,
                       cifar100=100,
                       cifar100_fine=100,
                       imagenet=1000,
                       rn50x16openai=1000,
                       vitb16laion400m=1000,
                       vitb16openai=1000,
                       aircraft=100,
                       birds=500,
                       caltech101=101,
                       caltech256=257,
                       cars=196,
                       dtd=47,
                       food101=101,
                       flowers=102,
                       pets=37,
                       SUN=397)[dataset_name]
    if model_name == 'ln5': return dict(mnist=LeNet5_MNIST, cifar10=LeNet5_CIFAR, cifar100=LeNet5_CIFAR).get(dataset_name, None)(num_classes=num_classes)
    if model_name == 'mn': return custom_mobilenet()
    if model_name == 'rn18': return resnet18(pretrained=False, num_classes=num_classes)
    if model_name == 'rn50': return resnet50(pretrained=False, num_classes=num_classes)
    if model_name == 'rn20': return resnet20(num_classes=num_classes)
    if model_name == 'rn32': return resnet32(num_classes=num_classes)
    if model_name == 'rn44': return resnet44(num_classes=num_classes)
    if model_name == 'rn56': return resnet56(num_classes=num_classes)
    if model_name == 'wrn-22-2': return Wide_ResNet(22, 2, num_classes=num_classes)
    if model_name == 'wrn-22-4': return Wide_ResNet(22, 4, num_classes=num_classes)
    if model_name == 'wrn-28-2': return Wide_ResNet(28, 2, num_classes=num_classes)
    if model_name == 'wrn-28-4': return Wide_ResNet(28, 4, num_classes=num_classes)
    if model_name == 'wrn-28-8': return Wide_ResNet(28, 8, num_classes=num_classes)
    if model_name == 'wrn-40-2': return Wide_ResNet(40, 2, num_classes=num_classes)
    if model_name == 'wrn-16-4': return wide_resnet_16_4(num_classes=num_classes)
    if model_name == 'wrn-16-5': return wide_resnet_16_5(num_classes=num_classes)
    if model_name == 'wrn-16-6': return wide_resnet_16_6(num_classes=num_classes)
    if model_name == 'wrn-16-8': return wide_resnet_16_8(num_classes=num_classes)
    if model_name == 'wrn-40-4': return wide_resnet_40_4(num_classes=num_classes)
    if model_name == 'wrn-16-4-no-ws': return wide_resnet_no_ws_16_4(num_classes=num_classes)
    if model_name == 'wrn-16-4-sc': return wide_resnet_16_4_sc(num_classes=num_classes)
    if model_name == 'resnet9-sc': return resnet9_sc(num_classes=num_classes)
    if model_name == 'nf-rn-50': return nf_resnet_50(num_classes=num_classes)

    if model_name == 'cnn5': return CNN5(num_classes=num_classes)
    if model_name == 'cnn5-papernot': return CNN5_Papernot()
    if model_name == 'cnn2-papernot': return CNN2_Papernot()

    if model_name == 'deit_tiny': return timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    if model_name == 'deit_small': return timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=num_classes)
    if model_name == 'deit_base': return timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=num_classes)

    if model_name == 'vgg16': return timm.create_model('vgg16', pretrained=False, num_classes=num_classes)

    if 'vit' in model_name and 'cifar' in dataset_name:
        return vit_cifar(model_name)

    model_mapping = {
        'vit-t': 'vit_tiny_patch16_224', # 5_526_346 params
        'vit-s': 'vit_small_patch16_224', # 21_669_514 params
        'vit-b': 'vit_base_patch16_224', # 85_806_346 params
        'vit-b32': 'vit_base_patch32_224', # 87_532_132
        'vit-l': 'vit_large_patch16_224', # 303_311_882 params
        'rn10t': 'resnet10t', # 5_435_488 params
        'rn14t': 'resnet14t', # 10_081_632 params

    }
    if model_name in model_mapping:
        print(f'Loading pre-trained {model_name}: {pretrained}')
        return timm.create_model(model_mapping[model_name], pretrained=False, num_classes=num_classes)

    raise RuntimeError(f'Model {model_name} is not implemented!')

def get_optimizer(args, params):
    if args.optimizer == 'sgd':
        return SGD(
            params,
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        return Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,  
        )
    if args.optimizer == 'micro-adam':
        return MicroAdam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l2_norm_clip=args.dp_c,
            sigma=args.dp_sigma,
            batch_size=args.batch_size,
            betas=(0.9, 0.999),
            eps=1e-8,
            num_grads=args.micro_adam_num_grads,
            density=args.micro_adam_density,
            device=get_first_device())
    if args.optimizer == 'adambc':
        return AdamBC(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l2_norm_clip=args.dp_c,
            sigma=args.dp_sigma,
            batch_size=args.batch_size,
            betas=(0.9, 0.999),
            eps=1e-8,
            eps2=5e-8,
            device=get_first_device())
    if args.optimizer == 'dp-micro-adam':
        return DPMicroAdam_v1(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l2_norm_clip=args.dp_c,
            sigma=args.dp_sigma,
            batch_size=args.batch_size,
            betas=(0.9, 0.999),
            eps=1e-8,
            num_grads=args.micro_adam_num_grads,
            density=args.micro_adam_density,
            device=get_first_device())
    if args.optimizer == 'dp-sgd':
        return DPSGD(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l2_norm_clip=args.dp_c,
            sigma=args.dp_sigma,
            batch_size=args.batch_size,
            device=get_first_device())
    if args.optimizer == 'dp-micro-adam-stp':
        return DPMicroAdamStP(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l2_norm_clip=args.dp_c,
            sigma=args.dp_sigma,
            batch_size=args.batch_size,
            betas=(0.9, 0.999),
            eps=1e-8,
            eps2=1e-2,
            num_grads=args.micro_adam_num_grads,
            density=args.micro_adam_density,
            device=get_first_device())
    if args.optimizer == 'dp-adam-stp':
        return DPAdamStP(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l2_norm_clip=args.dp_c,
            sigma=args.dp_sigma,
            batch_size=args.batch_size,
            betas=(0.9, 0.999),
            eps=1e-8,
            eps2=1e-2,
            device=get_first_device())
    
    raise ValueError(f'Optimizer {args.optimizer} is not implemented!')


def get_gpu_mem_usage():
    """
        This method returns the GPU memory usage for the current process.
        It uses gpustat to query the GPU used by the current process (using CUDA_VISIBLE_DEVICES)

        GPUSTAT usage:
        stat = gpustat.new_query().gpus # this is a list containing information about each GPU indexed from 0 to 7
        stat[i] (GPU #i) has the following keys:
            - 'index'
            - 'uuid'
            - 'name'
            - 'temperature.gpu'
            - 'fan.speed'
            - 'utilization.gpu'
            - 'utilization.enc'
            - 'utilization.dec'
            - 'power.draw'
            - 'enforced.power.limit'
            - 'memory.used'
            - 'memory.total'
            - 'processes'
        Among these keys, only the key 'processes' is used here.
        stat[i].processes is a list of dicts, where each dict contains information about each process currently running on the GPU #i
            - 'username'
            - 'command'
            - 'full_command'
            - 'gpu_memory_usage'
            - 'cpu_percent'
            - 'cpu_memory_usage'
            - 'pid'
    """
    gpus = gpustat.new_query().gpus
    gids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    gpu_mem = sum([int(proc['gpu_memory_usage']) for gid in gids for proc in gpus[gid]['processes'] if int(proc['pid']) == os.getpid()])
    return gpu_mem

def get_first_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda:0')
