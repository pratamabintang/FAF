"""
TENT: Test-Time Adaptation via Entropy Minimization
Reference: https://github.com/DequanWang/tent
"""

import torch
import torch.nn as nn
import torch.jit

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing."""
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

    def forward(self, *args, **kwargs):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(*args, **kwargs)

        return outputs

    def reset(self):
        pass

    def forward_and_adapt(self, *args, **kwargs):
        """Forward and adapt model on batch of data."""
        self.optimizer.zero_grad()
        outputs = self.model(*args, **kwargs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        # Calculate softmax entropy loss
        loss = softmax_entropy(logits).mean()
        loss.backward()
        self.optimizer.step()
        return outputs

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def collect_params(model):
    """Walk through model and collect affine / norm parameters."""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model):
    """Configure model for use with tent."""
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model
