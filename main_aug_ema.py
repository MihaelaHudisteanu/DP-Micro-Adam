import warnings; warnings.filterwarnings("ignore")
from helpers.args import get_arg_parse

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split, Sampler
from tqdm import tqdm
import math
import wandb
from datetime import datetime
import psutil

from helpers.args import *
from helpers.datasets import *
from helpers.lr_scheduling import *
from helpers.tools import *
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import RDPAccountant
from helpers.augmult import AugmultConfig
from helpers.param_averaging import EMA

torch.backends.cuda.matmul.allow_tf32 = True

def get_num_steps(eps, delta, noise_multiplier, sampling_rate):
    accountant = RDPAccountant()
    current_eps = 0
    num_steps = 0

    while current_eps < eps:
        num_steps += 1
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sampling_rate)
        current_eps = accountant.get_epsilon(delta=delta)
    return num_steps

class PoissonBatchSampler(Sampler):
    def __init__(self, n_samples, q: float, steps_per_epoch: int, generator=None):
        self.n_samples = n_samples
        self.q = q
        self.steps_per_epoch = steps_per_epoch
        self.gen = generator
        self.indices = torch.arange(n_samples)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            mask = torch.rand(self.n_samples, generator=self.gen) < self.q
            idx = self.indices[mask]
            if idx.numel() == 0:
                continue  # skip empty batch
            yield idx.tolist()

    def __len__(self):
        return self.steps_per_epoch

@torch.no_grad()
def test(model, data):
    loss, correct, test_dataset_size = 0, 0, 0
    with torch.inference_mode():
        for x, y in tqdm(data):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            y_hat = model(x)
            test_dataset_size += x.shape[0]

            loss += cross_entropy(y_hat, y, reduction='sum').item()
            correct += (y_hat.argmax(1) == y).sum().item()
    return correct, loss, test_dataset_size

def main():
    args = get_arg_parse()
    set_all_seeds(args.seed)

    if args.wandb_enable:
        setup_wandb(args.wandb_project, args.wandb_job_type, args.wandb_group, args.wandb_name, args)

    model = GradSampleModule(get_model(args).to(args.device))
    optimizer = get_optimizer(args, model.parameters())
    train_data, test_data = get_datasets(args.dataset_name, args.dataset_path)

    if args.dataset_name == 'imagenet':
        val_len = 10_000
        train_len = len(train_data) - val_len
    if args.dataset_name in {'cifar10', 'cifar10_fine', 'cifar100', 'cifar100_fine', 'mnist'}:
        val_ratio = 0.1
        train_len = int((1 - val_ratio) * len(train_data))
        val_len = len(train_data) - train_len
    train_data, val_data = random_split(train_data, [train_len, val_len], torch.Generator().manual_seed(args.seed))  

    iterations = 0
    n_samples = len(train_data)
    q = args.batch_size / n_samples
    steps_per_epoch = math.ceil(n_samples / args.batch_size)
    total_steps = get_num_steps(args.dp_eps, args.dp_delta, args.dp_sigma, q)

    if args.numeric_aprox:
        total_steps = int(args.epochs / q)
        args.dp_sigma = get_noise_multiplier(
            target_epsilon=args.dp_eps,
            target_delta=args.dp_delta,
            sample_rate=q,
            steps=total_steps
        )
        if args.wandb_enable:
            wandb.config.update({'dp_sigma': args.dp_sigma}, allow_val_change=True)

    args.lr_warmup_steps = total_steps * args.lr_warmup_steps

    batch_sampler = PoissonBatchSampler(n_samples, q, steps_per_epoch, torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(
        train_data,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_data, shuffle=False, batch_size=128, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=128, num_workers=2, pin_memory=True)

    microbatch_size = args.micro_batch_size
    accountant = RDPAccountant()

    config = AugmultConfig(augmult=args.aug_mult)  # uses DeepMind defaults
    if args.ema:
        ema = EMA(model, decay=args.ema, use_warmup=True, start_step=0)

    time_start_training = datetime.now()
    epoch = 0
    current_eps = 0
    while current_eps < args.dp_eps:
        time_start_train_epoch = datetime.now()
        progress_bar = tqdm(train_loader)
        train_loss, train_size, train_correct = 0., 0, 0
        
        for x, y in progress_bar:
            iterations += 1
            optimizer.zero_grad()

            for i in range(0, x.size(0), microbatch_size):
                xb = x[i:i+microbatch_size].cuda(non_blocking=True)
                yb = y[i:i+microbatch_size].cuda(non_blocking=True)

                # eval
                with torch.inference_mode():
                    y_hat = model(xb)
                    loss = cross_entropy(y_hat, yb)

                train_size += xb.shape[0]
                train_loss += loss.item() * xb.size(0)  
                train_correct += (torch.argmax(y_hat, 1) == yb).sum().item()

                # augmentation
                B = xb.size(0)
                K = max(config.augmult, 1)

                x_aug, y_aug =  config.apply(xb, yb)
                y_hat_aug = model(x_aug)
                per_view = cross_entropy(y_hat_aug, y_aug, reduction="none")
                loss = per_view.view(K, B).mean(dim=0).mean()

                loss.backward()
                optimizer.average_augmented_grads(model, B, K)  # type: ignore

                optimizer.microbatch_step() # type: ignore

            # end microbatch
            optimizer.step()
            accountant.step(noise_multiplier=args.dp_sigma, sample_rate=q)
            if args.ema:
                ema.update(model)

            lr = update_lr(sched=args.lr_sched,
                           optimizer=optimizer,
                           base_lr=args.base_lr,
                           step=iterations,
                           warmup_steps=args.lr_warmup_steps,
                           total_steps=total_steps)

            if args.wandb_enable:
                wandb.log({f'step': iterations, f'step/step': iterations, f'step/lr': lr})

        # end for x, y
        now = datetime.now()
        time_elapsed_training = now - time_start_training
        time_elapsed_train_epoch = now - time_start_train_epoch

        ##### MODEL EVALUATION
        train_loss /= train_size
        train_accuracy = train_correct / train_size

        val_correct, val_loss, val_size = test(model, val_loader)
        val_loss /= val_size
        val_accuracy = val_correct / val_size

        if args.ema:
            with ema.average_parameters(model):
                ema_val_correct, ema_val_loss, ema_val_size = test(model, val_loader)
                ema_val_loss /= ema_val_size
                ema_val_accuracy = ema_val_correct / ema_val_size

        current_eps = accountant.get_epsilon(delta=args.dp_delta)
        epoch += 1
        if args.wandb_enable:
            wandb.config.update({'epochs': epoch}, allow_val_change=True)

        if args.wandb_enable:
            wandb_data = {
                'epoch/step': iterations,
                'epoch/epoch': epoch,
                'epoch/eps': current_eps,
                'epoch/train_acc': train_accuracy,
                'epoch/train_loss': train_loss,
                'epoch/val_acc': val_accuracy,
                'epoch/val_loss': val_loss,
                'epoch/epoch_time': time_elapsed_train_epoch.total_seconds(),
                'epoch/total_elapsed': time_elapsed_training.total_seconds(),
                'epoch/lr': lr,
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
            test_correct, test_loss, test_size = test(model, test_loader)
    else:
        test_correct, test_loss, test_size = test(model, test_loader)
    
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

if __name__ == '__main__':
    main()
