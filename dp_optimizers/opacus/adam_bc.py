# bias correction see https://arxiv.org/pdf/2312.14334
import torch

from dp_optimizers.shared import *

class AdamBC(torch.optim.Optimizer):
    def __init__(self,
                 params, lr, weight_decay, l2_norm_clip, sigma, batch_size,
                 betas=(0.9, 0.999), eps=1e-8, eps2=5e-8, device='cuda'):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        self.lr = lr
        self.weight_decay = weight_decay
        self.l2_norm_clip = l2_norm_clip
        self.sigma = sigma
        self.batch_size = batch_size

        self.optim_steps = 0
        self.betas = betas
        self.eps = eps
        self.eps2 = eps2
        self.device = device

        self.init_optimizer_states()

    def loop_params(self, check_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                if check_grad:
                    if p.grad is None: continue
                yield group, self.state[p], p


    @torch.no_grad()
    def init_optimizer_states(self):
        for group, state, p in self.loop_params(check_grad=False):
            state[STATE_MT] = torch.zeros_like(p)
            state[STATE_VT] = torch.zeros_like(p)
            # print(f'Initialized gradient buffer for parameter {id(p)}')

    @torch.no_grad()
    def step(self, closure=None):
        self.optim_steps += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        t = self.optim_steps
        beta1, beta2 = self.betas
        eps = self.eps
        eps2 = self.eps2
        bias = self.sigma * self.l2_norm_clip / self.batch_size

        for group, state, p in self.loop_params():
            state = self.state[p]
            wd = group.get('weight_decay', self.weight_decay)
            lr = group['lr']
    
            mt = state[STATE_MT]
            vt = state[STATE_VT]
            g = p.grad   
    
            mt.mul_(beta1).add_(g, alpha=1.0 - beta1)
            vt.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            
            m_hat = mt / (1 - beta1 ** t)
            v_hat = vt / (1 - beta2 ** t)

            v_hat = torch.clamp(v_hat - bias ** 2, min=eps2)
    
            if wd > 0: p.mul_(1 - lr * wd)
            p.add_(m_hat / (eps + v_hat.sqrt()), alpha=-lr)
        
        return loss