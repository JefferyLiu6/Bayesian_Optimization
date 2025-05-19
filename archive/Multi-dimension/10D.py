import torch
import gpytorch
import botorch
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("bmh")

from tqdm.notebook import tqdm

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

def visualize_gp_belief_and_policy(model, likelihood, policy=None, next_x=None):
    # Disabled for 10D Levy function; kept for compatibility
    pass

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

def fit_gp_model(train_x, train_y, num_train_iters=500):
    # Declare the GP
    noise = 1e-4

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.likelihood.noise = noise

    # Train the hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    for i in tqdm(range(num_train_iters), desc="Training GP"):
        optimizer.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_y)

        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood

# Initialize Levy function (10D) and bounds
levy = Levy(dim=10)
bounds = levy.bounds  # [[-10, ..., -10], [10, ..., 10]]

# Parameters
n_initial_points = 10
num_queries = 100
num_runs = 10

# Store minimum f(x) for each run
all_min_f_values = []

for run in range(num_runs):
    print(f"\nRun {run + 1}/{num_runs}")
    
    # Set different random seed for each run
    torch.manual_seed(2 + run)
    
    # Initialize with 10 random points in [-10, 10]^10
    train_x = torch.rand(size=(n_initial_points, 10)) * 20 - 10  # Scale to [-10, 10]
    train_y = levy(train_x)

    # Print initial points and their function values
    print("Initial points (x1, ..., x10, f(x)):")
    print(torch.hstack([train_x, train_y.unsqueeze(1)]))

    # Fit initial GP model
    model, likelihood = fit_gp_model(train_x, train_y)

    # Track minimum f(x) for this run
    min_f_values = [train_y.min().item()]

    # Run Bayesian optimization
    for i in range(num_queries):
        print(f"Iteration {i}, incumbent x=({train_x[train_y.argmin()].numpy().round(4).tolist()}), f(x)={train_y.min().item():.4f}")

        model, likelihood = fit_gp_model(train_x, train_y)

        policy = botorch.acquisition.analytic.ProbabilityOfImprovement(
            model, best_f=train_y.min(), maximize=False
        )
        
        next_x, acq_val = botorch.optim.optimize_acqf(
            policy,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=50,
        )

        next_y = levy(next_x)

        # Ensure next_y is 1D with shape (1,)
        next_y = next_y.unsqueeze(0) if next_y.dim() == 0 else next_y

        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])
        
        # Store the minimum f(x)
        min_f_values.append(train_y.min().item())

    # Store results for this run
    all_min_f_values.append(min_f_values)

    # Plot minimum f(x) vs iteration for this run
    plt.figure(figsize=(8, 6))
    plt.plot(range(num_queries + 1), min_f_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Minimum f(x)')
    plt.title(f'Minimum f(x) vs Iteration (Run {run + 1})')
    plt.grid(True)
    plt.savefig(f'min_f_vs_iteration_run_{run + 1}.png')
    plt.close()

# Convert to numpy array for easier computation
all_min_f_values = np.array(all_min_f_values)  # Shape: (num_runs, num_queries + 1)

# Compute mean, std, and 95% CI
mean_f = np.mean(all_min_f_values, axis=0)
std_f = np.std(all_min_f_values, axis=0)
ci_95 = 1.96 * std_f / np.sqrt(num_runs)  # 95% CI: z=1.96 for normal distribution

# Plot convergence with mean, std, and CI
plt.figure(figsize=(10, 6))
iterations = range(num_queries + 1)

# Plot mean
plt.plot(iterations, mean_f, marker='o', linestyle='-', color='b', label='Mean Minimum f(x)')

# Plot 95% CI
plt.fill_between(
    iterations,
    mean_f - ci_95,
    mean_f + ci_95,
    color='b',
    alpha=0.2,
    label='95% Confidence Interval'
)

# Plot mean ± std
plt.fill_between(
    iterations,
    mean_f - std_f,
    mean_f + std_f,
    color='g',
    alpha=0.1,
    label='Mean ± Std'
)

plt.xlabel('Iteration')
plt.ylabel('Minimum f(x)')
plt.title('Convergence Plot: Mean, Std, and 95% CI over 10 Runs (Levy 10D)')
plt.grid(True)
plt.legend()
plt.savefig('convergence_plot.png')
plt.close()