import torch

class Forrester:
    """f(x) = (6x − 2)^2 sin(12x − 4),  x ∈ [0,1]"""
    def __init__(self):
        self.bounds = torch.tensor([[0.0], [1.0]])
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((6*x - 2)**2) * torch.sin(12*x - 4)

class Branin:
    """Branin-Hoo f(x1,x2) on [−5,10]×[0,15]"""
    def __init__(self):
        self.bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])
        # recommended constants
        self.a, self.b = 1.0, 5.1/(4*torch.pi**2)
        self.c, self.r = 5.0/torch.pi, 6.0
        self.s, self.t = 10.0, 1.0/(8*torch.pi)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[...,0], x[...,1]
        term1 = self.a*(x2 - self.b*x1**2 + self.c*x1 - self.r)**2
        term2 = self.s*(1 - self.t)*torch.cos(x1) + self.s
        return (term1 + term2).squeeze(-1)

class Rosenbrock:
    """Rosenbrock (banana) f(x) = ∑ [100(x_{i+1}-x_i^2)^2 + (x_i - 1)^2]"""
    def __init__(self, dim=2):
        self.dim = dim
        # usually evaluated on [−5,10]^d
        self.bounds = torch.stack([ -5*torch.ones(dim),
                                     10*torch.ones(dim) ])
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        xi, xnext = x[..., :-1], x[..., 1:]
        return (100*(xnext - xi**2)**2 + (xi - 1)**2).sum(dim=-1)

class Hartmann:
    """Generic Hartmann in 3,4,6 dims on [0,1]^d."""
    _params = {
        3: {
            "alpha": torch.tensor([1.0,1.2,3.0,3.2]),
            "A":     torch.tensor([[3.0,10,30],
                                   [0.1,10,35],
                                   [3.0,10,30],
                                   [0.1,10,35]]),
            "P":     torch.tensor([[3689,1170,2673],
                                   [4699,4387,7470],
                                   [1091,8732,5547],
                                   [381,5743,8828]]) * 1e-4
        },
        4: {
            "alpha": torch.tensor([1.0,1.2,3.0,3.2]),
            "A":     torch.tensor([[10,3,17,3.5],
                                   [0.05,10,17,0.1],
                                   [3,3.5,1.7,10],
                                   [17,8,0.05,10]]),
            "P":     torch.tensor([[1312,1696,5569,124],
                                   [2329,4135,8307,3736],
                                   [2348,1451,3522,2883],
                                   [4047,8828,8732,5743]]) * 1e-4
        },
        6: {
            "alpha": torch.tensor([1.0,1.2,3.0,3.2]),
            "A":     torch.tensor([[10,3,17,3.5,1.7,8],
                                   [0.05,10,17,0.1,8,14],
                                   [3,3.5,1.7,10,17,8],
                                   [17,8,0.05,10,0.1,14]]),
            "P":     torch.tensor([[1312,1696,5569,124,8283,5886],
                                   [2329,4135,8307,3736,1004,9991],
                                   [2348,1451,3522,2883,3047,6650],
                                   [4047,8828,8732,5743,1091,381]]) * 1e-4
        }
    }

    def __init__(self, dim=3):
        assert dim in (3,4,6)
        p = self._params[dim]
        self.alpha, self.A, self.P = p["alpha"], p["A"], p["P"]
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        d = x.size(-1)
        inner = ((x.unsqueeze(-2) - self.P)**2 * self.A).sum(dim=-1)  # (..., 4)
        return -(self.alpha * torch.exp(-inner)).sum(dim=-1)
    


class Levy:
    """Levy function in 10 dimensions on [-10,10]^10."""
    def __init__(self, dim=10):
        self.dim = dim
        self.bounds = torch.stack([torch.full((dim,), -10.0), torch.full((dim,), 10.0)])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
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
        
        # Total function value
        return term1 + term2 + term3