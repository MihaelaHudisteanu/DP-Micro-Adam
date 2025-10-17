import torch
import math

from dp_optimizers import DPOptimizer
from dp_optimizers.shared import *


class DPMicroAdam(DPOptimizer):
    def __init__(self,
                 params, lr, weight_decay, C, sigma, batch_size,
                 *,
                 betas=(0.9, 0.999), eps=1e-8, num_grads=10, density=0.01):
        super().__init__(params, lr, weight_decay, C, sigma, batch_size)

        self.betas = betas
        self.eps = eps
        self.num_grads = num_grads
        self.density = density

        self.init_optimizer_states()

    @torch.no_grad()
    def init_optimizer_states(self):
        for group, state, p, ind in self.loop_params(check_grad=False):
            state[STATE_BUFFER_INDEX] = 0
            state[STATE_GRADIENT_DENSITY] = math.ceil(self.density * p.numel())
            state[STATE_EF] = torch.zeros_like(p)
            state[STATE_BUFFER_INDICES] = torch.zeros(self.num_grads, state[STATE_GRADIENT_DENSITY], dtype=torch.int32, device=p.device)
            state[STATE_BUFFER_VALUES] = torch.zeros(self.num_grads, state[STATE_GRADIENT_DENSITY], dtype=torch.float32, device=p.device)
            # print(f'Initialized gradient buffer for parameter {id(p)}')

    @torch.no_grad()
    def optimizer_update_step(self, closure=None):
        raise NotImplementedError

