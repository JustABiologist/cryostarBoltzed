import torch
from torch import nn
import torch.nn.functional as F


class RBM(nn.Module):

    def __init__(self, visible_dim: int, hidden_dim: int):
        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        # Parameters: weights and biases
        self.W = nn.Parameter(torch.empty(visible_dim, hidden_dim))
        self.visible_bias = nn.Parameter(torch.zeros(visible_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.W)

    def hidden_logits(self, v: torch.Tensor) -> torch.Tensor:
        return F.linear(v, self.W.t(), self.hidden_bias)

    def visible_logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.W, self.visible_bias)

    def sample_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        logits = self.hidden_logits(v)
        probs = torch.sigmoid(logits)
        with torch.no_grad():
            h = torch.bernoulli(probs)
        return h + (probs - probs.detach())

    def sample_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.visible_logits(h)
        probs = torch.sigmoid(logits)
        with torch.no_grad():
            v = torch.bernoulli(probs)
        return v + (probs - probs.detach())

    def gibbs_k(self, v0: torch.Tensor, k: int = 1) -> torch.Tensor:
        v = v0
        for _ in range(k):
            h = self.sample_h_given_v(v)
            v = self.sample_v_given_h(h)
        return v

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        # F(v) = -b^T v - sum_j log(1 + exp(c_j + (W^T v)_j))
        vbias = F.linear(v, torch.eye(self.visible_dim, device=v.device), self.visible_bias)
        hidden_term = F.softplus(self.hidden_logits(v)).sum(dim=-1)
        return -(vbias.sum(dim=-1)) - hidden_term

    def cd_loss(self, v_data: torch.Tensor, k: int = 1) -> torch.Tensor:
        # Contrastive divergence: F(v_data) - F(v_model)
        v_model = self.gibbs_k(v_data.detach(), k=k)
        return self.free_energy(v_data) - self.free_energy(v_model)


