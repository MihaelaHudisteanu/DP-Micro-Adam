import torch

class DPOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, l2_norm_clip, sigma, batch_size):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.lr = lr
        self.weight_decay = weight_decay
        self.l2_norm_clip = l2_norm_clip
        self.sigma = sigma
        self.batch_size = batch_size

        self.optim_steps = 0 # subscript t from the algorithm
        self.grad_count = 0 # how many gradients were computed (and implicitly clipped and saved to the buffer)

        self.params = [p for g in self.param_groups for p in g['params']]
        for g in self.param_groups:
            g['accum_grads'] = [torch.zeros_like(p, device=p.device) for p in g['params']]


    def zero_microbatch_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
            p.grad_sample = None

    @torch.no_grad()
    def average_augmented_grads(self, model, B, K):
        if K <= 1:
            return

        for param in self.params:
            if not param.requires_grad or param.grad is None:
                continue
            grad_samples = param.grad_sample  
            assert grad_samples.shape[0] == K * B, f'grad_sample has {grad_samples.shape[0]}, expected {K*B}'
            param.grad_sample = grad_samples.view(K, B, *grad_samples.shape[1:]).mean(dim=0)

    def microbatch_step(self):
        per_param_norms = []
        microbatch_size = 0

        # calculate the total norm of gradients
        for param in self.params:
            if param.requires_grad and param.grad is not None:
                grad_samples = param.grad_sample 

                microbatch_size = grad_samples.shape[0]

                grad_samples_flat = grad_samples.view(microbatch_size, -1)

                per_param_norms.append(grad_samples_flat.norm(2, dim=1) ** 2)

        total_norm_l2  = torch.stack(per_param_norms, dim=0).sum(dim=0).sqrt()
        clip_coef = (self.l2_norm_clip / (total_norm_l2 + 1e-6)).clamp(max=1.0)
        self.grad_count += microbatch_size 

        for group in self.param_groups:
            for param, accum_grad in zip(group['params'], group['accum_grads']):
                if not param.requires_grad or param.grad is None:
                    continue
                
                grad_samples = param.grad_sample
                clipped_grads = grad_samples * clip_coef.view(-1, *([1] * (grad_samples.dim() - 1)))

                # average across samples and accumulate
                accum_grad.add_(clipped_grads.sum(dim=0).to(accum_grad.device))
        
        self.zero_microbatch_grad()

    def loop_params(self, check_grad=True):
        for group in self.param_groups:
            for ind, p in enumerate(group['params']):
                if check_grad:
                    if p.grad is None: continue
                yield group, self.state[p], p, ind

    @torch.no_grad()
    def add_noise_then_divide(self):
        std = self.l2_norm_clip * self.sigma
        for g in self.param_groups:
            # generate all noise tensors in one list
            noises = [torch.randn(p.shape, device=accum_grad.device, dtype=accum_grad.dtype) * std for p, accum_grad in zip(g['params'], g['accum_grads'])]
            torch._foreach_add_(g['accum_grads'], noises)
            torch._foreach_div_(g['accum_grads'], self.grad_count)

    @torch.no_grad()
    def step(self, closure=None):
        """
            This method should be called at each step after the gradient for a single sample has been computed.
        """
        loss = None

        self.add_noise_then_divide()
        loss = self.optimizer_update_step(closure) 
        self.grad_count = 0 # reset counter

        return loss

    @torch.no_grad()
    def init_optimizer_states(self):
        raise NotImplementedError

    @torch.no_grad()
    def optimizer_update_step(self, closure=None):
        """
            This method implements the actual optimizer update step and should be implemented in the child classes.
            It will use the clipped gradient stored in the STATE_GRAD_BUFFER instead of raw gradient.
            IMPORTANT: this method supposes the noise was already added.
        """
        raise NotImplementedError
