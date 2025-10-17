import torch
from dp_optimizers.dp_micro_adam import DPMicroAdam
from dp_optimizers.shared import *


class DPMicroAdam_v1(DPMicroAdam):
    def __init__(self,
                 params, lr, weight_decay, l2_norm_clip, sigma, batch_size,
                 *,
                 betas=(0.9, 0.999), eps=1e-8, num_grads=10, density=0.01, device='cuda'):

        super().__init__(
            params, lr, weight_decay, l2_norm_clip, sigma, batch_size,
            betas=betas, eps=eps, num_grads=num_grads, density=density
        )
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

        ##### SHORTCUTS FOR HYPER-PARAMETERS
        t = self.optim_steps
        num_grads = self.num_grads
        beta1, beta2 = self.betas
        eps = self.eps

        for group, state, p, ind in self.loop_params():
            state = self.state[p]
            wd = group.get('weight_decay', self.weight_decay)
            lr = group['lr']
    
            ##### SHORTCUTS FOR STATES
            index = state[STATE_BUFFER_INDEX]
            k = state[STATE_GRADIENT_DENSITY]
            ef = state[STATE_EF]
            I = state[STATE_BUFFER_INDICES]
            V = state[STATE_BUFFER_VALUES]
            g = group['accum_grads'][ind]      
    
            ##### STEP 6: ERROR FEEDBACK
            a = g + ef
    
            ##### STEP 7: COMPUTE TOP-K
            indices, values = compute_standard_topk(a, k)
    
            ##### STEP 8: UPDATE BUFFER
            I[index, :] = indices
            V[index, :] = values
    
            ##### STEP 9: UPDATE INDEX
            state[STATE_BUFFER_INDEX] = (index + 1) % num_grads
    
            ##### STEP 10: UPDATE ERROR FEEDBACK
            a.view(-1)[indices] = 0 # remove "values" from accumulator; after that, what is left is the error feedback
    
            ##### STEP 11: SAVE SPARSE ACCUMULATOR TO ERROR FEEDBACK
            ef.zero_().add_(a)
    
            ##### STEPS 12+13: COMPUTE ADAM STATISTICS
            mt = g.zero_() # store first order momentum `m_t` in the gradient buffer to save memory
            vt = a.zero_() # store second order momentum `v_t` in the accumulator buffer to save memory
    
            for i in range(min(t, num_grads)):
                exp = (t - i - 1) % num_grads
                mt.view(-1)[I[i, :]] += (beta1 ** exp) * V[i, :]
                vt.view(-1)[I[i, :]] += (beta2 ** exp) * V[i, :].pow(2) 
    
            mt.mul_((1 - beta1) / (1 - beta1 ** t))
            vt.mul_((1 - beta2) / (1 - beta2 ** t))
    
            ##### STEP 14: UPDATE MODEL PARAMETERS
            if wd > 0: p.mul_(1 - lr * wd)
            p.add_(mt / (eps + vt.sqrt()), alpha=-lr)

        return loss
