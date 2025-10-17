import argparse
from helpers.tools import get_first_device

def get_arg_parse():
    parser = argparse.ArgumentParser()

    ##### WANDB
    parser.add_argument('--wandb_enable', type=int, choices=[0, 1], required=False, default=1, help='Whether to use wandb or not')
    parser.add_argument('--wandb_project', type=str, required=True, help='The wandb project inside "ist" owner.')
    parser.add_argument('--wandb_group', type=str, required=True, help='The wandb group in the project.')
    parser.add_argument('--wandb_job_type', type=str, default=None, required=True, help='The wandb job type')
    parser.add_argument('--wandb_name', type=str, required=True, default=None, help='The name for the experiment in wandb runs')

    ##### DATASET
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset to use for training.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name',
                        choices=[
                            'cifar10', 'cifar100',
                            'cifar10_fine', 'cifar100_fine',
                            'cifar10_no_aug',
                            'imagenet', 'imagenette', 'imagewoof','mnist'
                            # 'aircraft', 'birds', 'caltech101', 'caltech256', 'cars', 'dtd', 'flowers', 'food101', 'pets', 'SUN'
                        ])

    ##### MODEL
    parser.add_argument('--model_pretrained', type=int, choices=[0, 1], required=False, default=0, help='Whether to use a pretrained model or not')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=[
                            'vit-t', 'vit-s', 'vit-b', 'vit-l',
                            'rn18', 'rn20', 'rn32', 'rn44', 'rn50', 'rn56',
                            #'wrn-22-2', 'wrn-22-4', 'wrn-28-2', 'wrn-28-4', 'wrn-28-8', 'wrn-40-2', 
                            'wrn-16-4', 'wrn-16-5', 'wrn-16-6', 'wrn-16-8', 'wrn-40-4',
                            'wrn-16-4-no-ws',
                            'resnet9-sc','wrn-16-4-sc', 
                            'nf-rn-50',
                            'cnn5', 'cnn5-papernot', 'cnn2-papernot',
                            'deit_tiny', 'deit_small', 'deit_base'])

    ### OPTIMIZER
    parser.add_argument('--optimizer', required=True, help='Type of optimizer to use for training.',
                        choices=[
                            'sgd', 'adam', 'micro-adam', 'adambc',  # need to be wraped with the opacus GradSampleModule
                            'dp-micro-adam', 'dp-sgd',
                            'dp-micro-adam-stp', 'dp-adam-stp'])

    ### OTHER PARAMETERS
    parser.add_argument('--output_folder', type=str, required=True, help='Name of the file where the checkpoint of the most recent epoch is persisted.')
    parser.add_argument('--seed', type=int, required=True, help='The seed used to initialize the random number generator')

    ### GENERAL TRAINING PARAMETERS
    parser.add_argument('--epochs', type=int, required=True, help='The number of epochs to train the model for')
    parser.add_argument('--batch_size', type=int, required=True, help='Batchsize to use for training.')
    parser.add_argument('--micro_batch_size', type=int, default=128, required=False, help='Micro batchsize that accumulate into batchs.')
    parser.add_argument('--momentum', type=float, default=0, required=False, help='Momentum to use for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0, required=False, help='Weight decay to use for the optimizer.')
    parser.add_argument('--aug_mult', type=int, default=0, required=False, help='Augmentation multiplicity to use for training.')
    parser.add_argument('--ema', type=float, default=0, required=False, help='Decay for exponential moving average of model parameters.')

    ### LEARNING RATE
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--lr_warmup_steps', type=float, required=False, default=0, help='Number of epochs to warmup when --lr_units=epochs, otherwise steps')
    parser.add_argument('--lr_sched', type=str, required=True, help='The learning rate scheduler used for the optimizer',
                        choices=['cos', 'coswmp', 'linear', 'linearwmp', 'step', 'const'])

    ##### DP params
    parser.add_argument('--dp_c', type=float, required=True, help='Clipping constant')
    parser.add_argument('--dp_sigma', type=float, required=True, help='Noise magnitude')
    parser.add_argument('--numeric_aprox', type=int, choices=[0, 1], required=True, default=0, help='Whether to use numeric computation for noise, based on desired eps-delta')
    parser.add_argument('--dp_eps', type=float, required=False, help='Final epsilon')
    parser.add_argument('--dp_delta', type=float, required=False, help='Final delta')

    ##### DP-Micro-Adam params:
    parser.add_argument('--micro_adam_num_grads', type=int, required=True, help='Number of gradients for MicroAdam')
    parser.add_argument('--micro_adam_density', type=float, required=True, help='Gradient density for MicroAdam')

    return preprocess_args(parser.parse_args())


def preprocess_args(args):
    args.device = get_first_device()
    args.wandb_enable = bool(args.wandb_enable)
    args.model_pretrained = bool(args.model_pretrained)
    args.base_lr = args.lr
    return args
