import torch
import gpytorch
import botorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel_type="rbf"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type.lower() == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            base_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def build_gp(train_x: torch.Tensor, train_y: torch.Tensor, num_train_iters=50, kernel_type="rbf"):
    train_y = train_y.squeeze() if train_y.ndim > 1 else train_y

    likelihood = GaussianLikelihood()
    model = GPModel(train_x.double(), train_y.double(), likelihood, kernel_type)
    model.likelihood.noise = 1e-4

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=20,
        line_search_fn="strong_wolfe"
    )
    mll = ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    def closure():
        optimizer.zero_grad()
        output = model(train_x.double())
        loss = -mll(output, train_y.double())
        loss.backward()
        return loss

    for _ in tqdm(range(num_train_iters), desc="Training GP"):
        optimizer.step(closure)

    print(f"Lengthscale: {model.covar_module.base_kernel.lengthscale.item()}")
    print(f"Output scale: {model.covar_module.outputscale.item()}")
    print(f"Noise: {model.likelihood.noise.item()}")

    model.eval()
    likelihood.eval()

    return model, likelihood