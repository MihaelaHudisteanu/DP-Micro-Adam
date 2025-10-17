import numpy as np


def set_lr(optimizer, lr):
    if optimizer.__class__.__name__.startswith('Hook'):
        for name in optimizer.param_groups:
            optimizer.param_groups[name]['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def update_lr(sched, optimizer, base_lr, step, warmup_steps, total_steps, decay_at=None, step_rate=None, scaling_factor=1):
    sched = sched.lower()
    if sched == 'cos':
        lr = schedule_cosine(base_lr=base_lr, step=step, total_steps=total_steps)
    elif sched == 'coswmp':
        lr = schedule_cosine_warmup(base_lr=base_lr, step=step, warmup_steps=warmup_steps, total_steps=total_steps)
    elif sched == 'step':
        assert decay_at is not None, "Argument decay_at cannot be None"
        assert step_rate is not None, "Argument step_rate cannot be None"
        lr = schedule_multistep(base_lr=base_lr, step=step, decay_at_steps=decay_at, rate=step_rate)
    elif sched == 'linear':
        lr = schedule_linear(base_lr=base_lr, step=step, total_steps=total_steps)
    elif sched == 'linearwmp':
        lr = schedule_linear_warmup(base_lr=base_lr, step=step, warmup_steps=warmup_steps, total_steps=total_steps)
    elif sched == 'const':
        lr = base_lr
    else:
        raise RuntimeError(f'Learning rate schedule "{sched}" is not implemented!')
    set_lr(optimizer, lr / scaling_factor)
    return lr


def schedule_linear_warmup(base_lr, step, warmup_steps, total_steps):
    xs = [0, warmup_steps, total_steps]
    ys = [1e-4 * base_lr, base_lr, 0]
    return np.interp([step], xs, ys)[0]


def schedule_linear(base_lr, step, total_steps):
    return base_lr * (1 - step / total_steps)


def schedule_cosine(base_lr, step, total_steps):
    return 0.5 * (1 + np.cos(np.pi * step / total_steps)) * base_lr


def schedule_cosine_warmup(base_lr, step, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return schedule_cosine(base_lr, step=step-warmup_steps, total_steps=total_steps-warmup_steps)


def schedule_multistep(base_lr, step, decay_at_steps, rate):
    count = sum([int(step >= decay_step) for decay_step in decay_at_steps])
    return base_lr * (rate ** count)


# ##### SCHEDULES CALLABLE PER EPOCH
# def lr_sched_warmup_per_epoch(base_lr, warmup_length, epoch):
#     return base_lr * (epoch + 1) / warmup_length
#
#
# def lr_sched_cos_w_warmup_per_epoch(base_lr, epoch, epochs, warmup_length):
#     if epoch < warmup_length:
#         lr = lr_sched_warmup_per_epoch(base_lr, warmup_length, epoch)
#     else:
#         e = epoch - warmup_length
#         es = epochs - warmup_length
#         lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
#     return lr
