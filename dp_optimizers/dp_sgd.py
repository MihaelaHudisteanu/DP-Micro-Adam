import torch

from dp_optimizers import DPOptimizer
from dp_optimizers.shared import *


class DPSGD(DPOptimizer):
    def __init__(self,
                 params, lr, weight_decay, l2_norm_clip, sigma, batch_size,
                 *,
                 device='cuda'):
        super().__init__(params, lr, weight_decay, l2_norm_clip, sigma, batch_size)

        self.device = device
        for group in self.param_groups:
            group['accum_grads'] = [torch.zeros_like(param.data, device=device) if param.requires_grad else None for param in group['params']]

    @torch.no_grad()
    def optimizer_update_step(self, closure=None):
        self.optim_steps += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        t = self.optim_steps

        for group, state, p, ind in self.loop_params():
            state = self.state[p]
            wd = group.get('weight_decay', self.weight_decay)
            lr = group['lr']
    
            g = group['accum_grads'][ind]      
           
            if wd > 0: p.mul_(1 - lr * wd)
            p.add_(g, alpha=-lr)

        return loss