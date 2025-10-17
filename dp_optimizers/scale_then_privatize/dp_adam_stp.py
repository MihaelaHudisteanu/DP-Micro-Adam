import torch
from dp_optimizers.scale_then_privatize.dp_optimizer_stp import DPOptimizerStP
from dp_optimizers.shared import *


class DPAdamStP(DPOptimizerStP):
    def __init__(self,
                params, lr, weight_decay, l2_norm_clip, sigma, batch_size,
                *,
                betas=(0.9, 0.999), eps=1e-8, eps2=1e-2, device='cuda'):
        super().__init__(params, lr, weight_decay, l2_norm_clip, sigma, batch_size)

        self.betas = betas
        self.eps = eps
        self.eps2 = eps2
        self.device = device

        for group in self.param_groups:
            group['accum_grads'] = [
                torch.zeros_like(p.data, device=device) if p.requires_grad else None
                for p in group['params']
            ]
            group['s_t'] = [
                torch.ones_like(p.data, device=device) if p.requires_grad else None
                for p in group['params']
            ]
            group['m_t'] = [
                torch.zeros_like(p.data, device=device) if p.requires_grad else None
                for p in group['params']
            ]
            group['v_t'] = [
                torch.zeros_like(p.data, device=device) if p.requires_grad else None
                for p in group['params']
            ]

    @torch.no_grad()
    def optimizer_update_step(self, closure=None):
        self.optim_steps += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        t = self.optim_steps
        beta1, beta2 = self.betas
        eps = self.eps
        eps2 = self.eps2

        for group, state, p, ind in self.loop_params():
            g = group['accum_grads'][ind]

            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay)

            mt = group['m_t'][ind]
            vt = group['v_t'][ind]

            # Adam moments
            mt.mul_(beta1).add_(g, alpha=1.0 - beta1)
            vt.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

            # Bias corrections
            m_hat = mt / (1 - beta1 ** t)
            v_hat = vt / (1 - beta2 ** t)

            group['s_t'][ind] = 1.0 / (v_hat.sqrt() + eps2)

            if wd > 0.0: p.mul_(1.0 - lr * wd)
            # Parameter update
            p.add_(m_hat / (eps + v_hat.sqrt()), alpha=-lr)

        return loss
