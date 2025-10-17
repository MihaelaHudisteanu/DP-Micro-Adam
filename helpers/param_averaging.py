import contextlib
import torch

class EMA:
    def __init__(self, model, decay=0.9999, use_warmup=True, start_step=0):
        self.decay = decay
        self.use_warmup = use_warmup
        self.start_step = start_step
        self.num_updates = 0

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    @staticmethod
    def _schedule_mu(mu_base, t, use_warmup):
        if use_warmup:
            mu_warm = (1.0 + t) / (10.0 + t)
            if mu_warm < mu_base:
                return mu_warm
        return mu_base

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        t = max(self.num_updates - self.start_step, 0)

        if t == 0 and self.num_updates <= self.start_step:
            for s, p in zip(self.shadow, self.params):
                s.copy_(p.detach())
            return

        mu = self._schedule_mu(self.decay, t, self.use_warmup)

        torch._foreach_mul_(self.shadow, mu)
        torch._foreach_add_(self.shadow, self.params, alpha=1.0 - mu)


    @contextlib.contextmanager
    def average_parameters(self, model): 
        # swap model params with EMA copy (for eval)
        with torch.no_grad():
            backup = [p.detach().clone().requires_grad_(False) for p in self.params]
            
            for p, s in zip(self.params, self.shadow):
                p.copy_(s)

        try:
            yield
        finally:
            # restore original params
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)