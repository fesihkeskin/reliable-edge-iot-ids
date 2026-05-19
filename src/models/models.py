import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from scipy.optimize import minimize


class TinyMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=96, hidden2=48, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs, y_true):
        probs = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
        y_true = np.asarray(y_true, dtype=int)
        logits = np.log(probs)

        def nll(tarr):
            temperature = max(float(tarr[0]), 1e-6)
            z = logits / temperature
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p = p / p.sum(axis=1, keepdims=True)
            return -float(np.mean(np.log(p[np.arange(len(y_true)), y_true] + 1e-12)))

        result = minimize(nll, x0=np.array([1.0]), method="L-BFGS-B", bounds=[(1e-2, 100)])
        self.temperature = float(result.x[0]) if result.success else 1.0
        return self

    def transform(self, probs):
        probs = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
        z = np.log(probs) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


def clone_tiny_mlp(model, input_dim, output_dim):
    cloned = TinyMLP(input_dim, output_dim)
    cloned.load_state_dict({key: val.detach().cpu().clone() for key, val in model.state_dict().items()})
    return cloned


def apply_structured_pruning(model, amount=0.30):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    return model


def quantize_dynamic_cpu(model):
    return torch.quantization.quantize_dynamic(model.to("cpu").eval(), {nn.Linear}, dtype=torch.qint8)
