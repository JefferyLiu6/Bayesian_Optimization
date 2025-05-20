import math
import torch


# 1-D Forrester

class Forrester:
    r"""f(x)= (6x−2)² sin(12x−4)  on  x∈[0,1]."""
    bounds = torch.tensor([[0.0], [1.0]])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:          # (...,1) → (...)
        x = x.squeeze(-1)
        return ((6.0 * x - 2.0) ** 2 * torch.sin(12.0 * x - 4.0))


# 2-D Branin

class Branin:
    r"""Branin–Hoo on [−5,10]×[0,15].  Global minimum ≈ 0.397."""
    bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])

    a, b = 1.0, 5.1 / (4 * math.pi ** 2)
    c, r = 5.0 / math.pi, 6.0
    s, t = 10.0, 1.0 / (8 * math.pi)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:          # (...,2) → (...)
        x1, x2 = x[..., 0], x[..., 1]
        term1 = self.a * (x2 - self.b * x1 ** 2 + self.c * x1 - self.r) ** 2
        term2 = self.s * (1.0 - self.t) * torch.cos(x1) + self.s
        return term1 + term2


# 3D, 4D, 6D Hartmann
class Hartmann:
    """Hartmann in 3, 4, or 6 dimensions on [0,1]^d. Standard BO benchmark."""
    _params = {
        3: {
            "alpha": torch.tensor([1.0, 1.2, 3.0, 3.2]),
            "A": torch.tensor([[3.0, 10, 30],
                               [0.1, 10, 35],
                               [3.0, 10, 30],
                               [0.1, 10, 35]]),
            "P": torch.tensor([[3689, 1170, 2673],
                               [4699, 4387, 7470],
                               [1091, 8732, 5547],
                               [381, 5743, 8828]]) * 1e-4
        },
        4: {
            "alpha": torch.tensor([1.0, 1.2, 3.0, 3.2]),
            "A": torch.tensor([[10, 3, 17, 3.5],
                               [0.05, 10, 17, 0.1],
                               [3, 3.5, 1.7, 10],
                               [17, 8, 0.05, 10]]),
            "P": torch.tensor([[1312, 1696, 5569, 124],
                               [2329, 4135, 8307, 3736],
                               [2348, 1451, 3522, 2883],
                               [4047, 8828, 8732, 5743]]) * 1e-4
        },
        6: {
            "alpha": torch.tensor([1.0, 1.2, 3.0, 3.2]),
            "A": torch.tensor([[10, 3, 17, 3.5, 1.7, 8],
                               [0.05, 10, 17, 0.1, 8, 14],
                               [3, 3.5, 1.7, 10, 17, 8],
                               [17, 8, 0.05, 10, 0.1, 14]]),
            "P": torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]]) * 1e-4
        }
    }

    def __init__(self, dim=3):
        assert dim in (3, 4, 6), f"Dimension {dim} not supported; use 3, 4, or 6"
        p = self._params[dim]
        self.alpha, self.A, self.P = p["alpha"], p["A"], p["P"]
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:          # (...,d) → (...)
        d = x.size(-1)
        inner = ((x.unsqueeze(-2) - self.P) ** 2 * self.A).sum(dim=-1)  # (..., 4)
        return -(self.alpha * torch.exp(-inner)).sum(dim=-1)  # Negated for minimization



# 10D Levy

class Levy:
    """Levy function in 10 dimensions on [-10,10]^10."""
    def __init__(self, dim=10):
        self.dim = dim
        self.bounds = torch.stack([torch.full((dim,), -10.0), torch.full((dim,), 10.0)])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:          # (...,10) → (...)
        d = x.size(-1)
        assert d == self.dim, f"Expected dimension {self.dim}, got {d}"
        
        # Compute w_i = 1 + (x_i - 1) / 4
        w = 1 + (x - 1) / 4
        
        # First term: sin^2(pi w_1)
        term1 = torch.sin(torch.pi * w[..., 0]) ** 2
        
        # Middle terms: sum from i=1 to d-1 of (w_i - 1)^2 [1 + 10 sin^2(pi w_i + 1)]
        term2 = torch.sum(
            (w[..., :-1] - 1) ** 2 * (1 + 10 * torch.sin(torch.pi * w[..., :-1] + 1) ** 2),
            dim=-1
        )
        
        # Last term: (w_d - 1)^2 [1 + sin^2(2 pi w_d)]
        term3 = (w[..., -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[..., -1]) ** 2)
        
        return term1 + term2 + term3