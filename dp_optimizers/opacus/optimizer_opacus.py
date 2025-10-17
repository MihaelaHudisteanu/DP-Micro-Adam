import torch

class OtimizerOpacus(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, l2_norm_clip, sigma, batch_size):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.lr = lr
        self.weight_decay = weight_decay
        self.l2_norm_clip = l2_norm_clip
        self.sigma = sigma
        self.batch_size = batch_size

        self.optim_steps = 0 # subscript t from the algorithm
        self.grad_count = 0 # how many gradients were computed (and implicitly clipped and saved to the buffer)

    
    def loop_params(self, check_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                if check_grad:
                    if p.grad is None: continue
                yield group, self.state[p], p


    @torch.no_grad()
    def step(self, closure=None):
        """
            This method should be called at each step after the gradient for a single sample has been computed.
        """
        loss = None
        loss = self.optimizer_update_step(closure) 

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
