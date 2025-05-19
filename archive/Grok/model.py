"""
GP helpers – aligned with reference scripts for better convergence.
"""

import torch
import gpytorch
import botorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def build_gp(train_x: torch.Tensor, train_y: torch.Tensor, bounds: torch.Tensor, num_train_iters=1000):
    """
    Returns a trained GP model and likelihood with normalized inputs.

    train_x : (N, d)
    train_y : (N,) or (N,1)
    bounds : (2, d) - lower and upper bounds for each dimension
    """
    # Ensure train_y is 1D with shape (N,)
    train_y = train_y.squeeze() if train_y.ndim > 1 else train_y

    # Normalize inputs to [0,1] based on bounds
    lb, ub = bounds[0], bounds[1]
    train_x_normalized = (train_x - lb) / (ub - lb)

    # Initialize likelihood and model
    noise = 1e-4
    likelihood = GaussianLikelihood()
    model = GPModel(train_x_normalized.double(), train_y.double(), likelihood)
    model.likelihood.noise = noise

    # Train the hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for _ in tqdm(range(num_train_iters), desc="Training GP"):
        optimizer.zero_grad()
        output = model(train_x_normalized.double())
        loss = -mll(output, train_y.double())
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood