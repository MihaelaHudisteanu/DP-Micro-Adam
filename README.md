# DP-MicroAdam

Implementation of **DP-MicroAdam**, a differentially private extension of [MicroAdam (Modoran et al., 2024)](https://arxiv.org/abs/2405.15593) that integrates standard privacy mechanisms into a sparsity-aware and memory-efficient update rule.

It supports large-scale private training and fine-tuning using various optimizers, models, and DP configurations.

## How to Run

To launch an experiment, modify the arguments in ['run-dp-micro-adam.py'](./run-dp-micro-adam.py) to set the dataset, model, optimizer, and DP parameters.  

## Available Configurations

### Datasets
Available datasets:
cifar10, cifar100, cifar10_fine, cifar100_fine,
cifar10_no_aug, imagenet, mnist
- **Normal training:** 'cifar10', 'cifar100', 'imagenet' (standard augmentations).  
- **No augmentation or used for AugMult:** 'cifar10_no_aug'.  
- **Fine-tuning:** 'cifar10_fine', 'cifar100_fine' (include resize and standard augmentation).

### Models
Supported architectures:
rn18, rn20, rn32, rn44, rn50, rn56,
resnet9-sc
wrn-16-4, wrn-16-5, wrn-16-6, wrn-16-8, wrn-40-4,
nf-rn-50,
deit_tiny (pretrained), deit_small (pretrained), deit_base (pretrained),
vit-t, vit-s, vit-b, vit-l

### Optimizers
Available optimizers:
sgd, adam, micro-adam, adambc,
dp-sgd, dp-micro-adam, dp-micro-adam-stp, dp-adam-stp

- **Already DP-compatible:** 'dp-*' optimizers (run with 'main.py' or 'main-aug-ema.py').  
- **Non-DP optimizers:** ('sgd', 'adam', 'micro-adam', 'adambc') wrap with **Opacus** -> run via 'main-opacus.py' or 'main-opacus-distrib.py'.


### Differential Privacy
You can configure privacy in two ways:

1. **Fixed noise** ('numeric_aprox=0'):  
   Set 'dp_sigma' and train until reaching the target privacy budget ('dp_eps', 'dp_delta').
2. **Fixed epochs** ('numeric_aprox=1'):  
   Specify 'dp_eps', 'dp_delta', and number of 'epochs' -> the code will approximate the required noise scale.

**Key arguments:**
| Argument | Description | Example |
|-----------|--------------|----------|
| 'dp_c' | Gradient clipping norm | '1.0' |
| 'dp_sigma' | Noise multiplier | '4.0' |
| 'dp_eps', 'dp_delta' | Target privacy budget | '(8, 1e-5)' |

### EMA and Learning Rate
- 'ema': Exponential Moving Average of weights (set to '0' to disable, or e.g. '0.99' for smoothing).  
- 'lr_sched': Learning rate schedule ('cos', 'coswmp', 'step', 'linear', 'linearwmp', 'const').  
- 'lr_warmup_steps': Fraction of total steps used for LR warmup.

### Logging
- 'wandb_enable': Enable or disable [Weights & Biases](https://wandb.ai) logging.  
  Configure 'wandb_project', 'wandb_group', 'wandb_name', and 'tools.py/setup_wandb/entity' for experiment tracking.
