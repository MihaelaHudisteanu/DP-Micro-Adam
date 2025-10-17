import warnings; warnings.filterwarnings("ignore")
from helpers.args import get_arg_parse

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from datetime import datetime
import psutil

from helpers.args import *
from helpers.datasets import *
from helpers.lr_scheduling import *
from helpers.tools import *
from opacus import GradSampleModule
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.optimizers import DistributedDPOptimizer
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader, DistributedSampler
from opacus.accountants import RDPAccountant
from helpers.param_averaging import EMA

import torch.distributed as dist
import torch.multiprocessing as mp

import os
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_num_steps(eps, delta, noise_multiplier, sampling_rate):
    accountant = RDPAccountant()
    current_eps = 0
    num_steps = 0

    while current_eps < eps:
        num_steps += 1
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sampling_rate)
        current_eps = accountant.get_epsilon(delta=delta)
    return num_steps
    
@torch.no_grad()
def distrib_test(model, data_loader, rank=0, world_size=1):
    local_correct, local_loss, local_size = 0, 0.0, 0

    for x, y in data_loader: #tqdm(data_loader, disable=(rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            x, y = x.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)

            y_hat = model(x)
        local_size += x.shape[0]

        local_loss += cross_entropy(y_hat, y, reduction='sum').item()
        local_correct += (y_hat.argmax(1) == y).sum().item()

    correct_tensor = torch.tensor(local_correct, device=rank)
    loss_tensor = torch.tensor(local_loss, device=rank)
    size_tensor = torch.tensor(local_size, device=rank)

    if world_size > 1:
        dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(size_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        return correct_tensor.item(), loss_tensor.item(), size_tensor.item()
    else:
        return None, None, None

def main(rank, world_size, args):
    log_file = f"rank_{rank}.log"
    with open(log_file, "a") as f:
        f.write(f"\n[PID {os.getpid()}] Hello from rank {rank}/{world_size}\n")

    setup(rank, world_size)
    set_all_seeds(args.seed)

    if args.wandb_enable and rank == 0:
        setup_wandb(args.wandb_project, args.wandb_job_type, args.wandb_group, args.wandb_name, args)

    model = get_model(args).to(rank)
    model = GradSampleModule(model)
    model = DPDDP(model)

    optimizer = get_optimizer(args, model.parameters())
    optimizer = DistributedDPOptimizer(
        optimizer=optimizer,
        noise_multiplier=args.dp_sigma,
        max_grad_norm=args.dp_c,
        expected_batch_size=args.batch_size//world_size,
    )
    train_data, test_data = get_datasets(args.dataset_name, args.dataset_path)

    if args.dataset_name == 'imagenet':
        val_len = 10_000
        train_len = len(train_data) - val_len
    if args.dataset_name in {'cifar10', 'cifar10_fine', 'cifar100', 'cifar100_fine', 'mnist'}:
        val_ratio = 0.1
        train_len = int((1 - val_ratio) * len(train_data))
        val_len = len(train_data) - train_len
    train_data, val_data = random_split(train_data, [train_len, val_len], torch.Generator().manual_seed(args.seed)) 

    n_samples = len(train_data)
    q = args.batch_size / n_samples

    train_loader = DPDataLoader.from_data_loader(
        data_loader=DataLoader(train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True),
        distributed=True,
    ) 
    val_loader = DataLoader(
        val_data,
        batch_size=128,
        num_workers=4, 
        pin_memory=True,
        sampler=DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        num_workers=4, 
        pin_memory=True,
        sampler=DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False),
    )

    ema = EMA(model, decay=0.9999, use_warmup=True, start_step=0)

    accountant = RDPAccountant()
    microbatch_size = args.micro_batch_size
    iterations = 0
    logical_iterations = 0

    if args.ema:
        ema = EMA(model, decay=args.ema, use_warmup=True, start_step=0)

    time_start_training = datetime.now()
    epoch = 0
    current_eps = 0
    while current_eps < args.dp_eps:
        time_start_train_epoch = datetime.now()
        with BatchMemoryManager(
            data_loader=train_loader, # type: ignore
            max_physical_batch_size=microbatch_size,
            optimizer=optimizer
        ) as microtrain_loader:
            train_loss, train_size, train_correct = 0., 0, 0
            
            for x, y in microtrain_loader:
                iterations += 1
                x, y = x.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
                
                optimizer.zero_grad()
                y_hat = model(x)
                loss = cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()

                crt_batch_size = x.shape[0]
                train_size += crt_batch_size
                step_loss = loss.item() * x.size(0)  # the loss for the current batch
                train_loss += step_loss  # the loss for the epoch
                train_correct += (torch.argmax(y_hat, 1) == y).sum().item()

                if not getattr(optimizer, "_is_last_step_skipped", False):
                    logical_iterations += 1 
                    accountant.step(noise_multiplier=args.dp_sigma, sample_rate=q)
                    if args.ema:
                        ema.update(model)
                    
                    if args.wandb_enable and rank == 0:
                        wandb.log({f'step': logical_iterations, f'step/step': logical_iterations, f'step/lr': args.base_lr})

            # end for x, y
        now = datetime.now()
        time_elapsed_training = now - time_start_training
        time_elapsed_train_epoch = now - time_start_train_epoch

        epoch += 1
        current_eps = accountant.get_epsilon(delta=args.dp_delta)
        
        ##### MODEL EVALUATION
        loss_tensor = torch.tensor(train_loss, device=rank)
        correct_tensor = torch.tensor(train_correct, device=rank)
        size_tensor = torch.tensor(train_size, device=rank)

        if world_size > 1:
            dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(size_tensor, dst=0, op=dist.ReduceOp.SUM)

        val_correct, val_loss, val_size = distrib_test(model, val_loader, rank, world_size)

        if args.ema:
            with ema.average_parameters(model):
                ema_val_correct, ema_val_loss, ema_val_size = distrib_test(model, val_loader, rank, world_size)

        dist.barrier()

        if rank == 0:
            train_loss = loss_tensor.item() / size_tensor.item()
            train_acc = correct_tensor.item() / size_tensor.item()

            val_loss /= val_size                  
            val_accuracy = val_correct / val_size 

            if args.ema:
                ema_val_loss /= ema_val_size                       
                ema_val_accuracy = ema_val_correct / ema_val_size  

            if args.wandb_enable:
                wandb.config.update({'epochs': epoch}, allow_val_change=True)

            if args.wandb_enable:
                wandb_data = {
                    'epoch/step': logical_iterations,
                    'epoch/epoch': epoch,
                    'epoch/eps': current_eps,
                    'epoch/train_acc': train_acc,
                    'epoch/train_loss': train_loss,
                    'epoch/val_acc': val_accuracy,
                    'epoch/val_loss': val_loss,
                    'epoch/epoch_time': time_elapsed_train_epoch.total_seconds(),
                    'epoch/total_elapsed': time_elapsed_training.total_seconds(),
                    'epoch/lr': args.base_lr,  # multistep learning rate is logged once per epoch
                    'epoch/ram_mem_usage': round(psutil.Process().memory_info().rss / (2 ** 30), 2),
                    #'epoch/gpu_mem_usage': get_gpu_mem_usage(),
                }
                wandb.log(wandb_data)

                if args.ema:
                    wandb_data = {
                        'epoch/ema_val_acc': ema_val_accuracy,
                        'epoch/ema_val_loss': ema_val_loss,
                    }
                    wandb.log(wandb_data)
        # end for epoch
    
    if args.ema:
        with ema.average_parameters(model):
            test_correct, test_loss, test_size = distrib_test(model, test_loader, rank, world_size)
    else:
        test_correct, test_loss, test_size = distrib_test(model, test_loader, rank, world_size)

    if rank == 0:
        test_accuracy = test_correct / test_size
        test_loss /= test_size

        if args.wandb_enable:
            wandb_data = {
                'final/test_acc': test_accuracy,
                'final/test_loss': test_loss,
                'final/eps': current_eps,
            }
            wandb.log(wandb_data)

    now = datetime.now()
    print(f'Training ended at {datetime.now()}, elapsed {now - time_start_training}')

    cleanup()

if __name__ == '__main__':
    args = get_arg_parse()
    world_size = torch.cuda.device_count()
    print(f'{world_size} gpus')
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True) # type: ignore