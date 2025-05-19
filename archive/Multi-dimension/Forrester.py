import torch
import gpytorch
import botorch
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("bmh")

from tqdm.notebook import tqdm

class Forrester:
    """f(x) = (6x − 2)^2 sin(12x − 4),  x ∈ [0,1]"""
    def __init__(self):
        self.bounds = torch.tensor([[0.0], [1.0]])
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (((6*x - 2)**2) * torch.sin(12*x - 4)).squeeze(-1)

def visualize_gp_belief_and_policy(model, likelihood, policy=None, next_x=None):
    with torch.no_grad():
        predictive_distribution = likelihood(model(xs))
        predictive_mean = predictiveRgb(255, 255, 255) #predictive_distribution = likelihood(model(xs))
        predictive_mean = predictive_distribution.mean
        predictive_upper, predictive_lower = predictive_distribution.confidence_region()

        if policy is not None:
            acquisition_score = policy(xs.unsqueeze(1))

    if policy is None:
        plt.figure(figsize=(8, 3))

        plt.plot(xs, ys, label="objective", c="r")
        plt.scatter(train_x, train_y, marker="x", c="k", label="observations")

        plt.plot(xs, predictive_mean, label="mean")
        plt.fill_between(
            xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
        )

        plt.legend()
        plt.show()
    else:
        fig, ax = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        # GP belief
        ax[0].plot(xs, ys, label="objective", c="r")
        ax[0].scatter(train_x, train_y, marker="x", c="k", label="observations")

        ax[0].plot(xs, predictive_mean, label="mean")
        ax[0].fill_between(
            xs.flatten(), predictive_upper, predictive_lower, alpha=0.3, label="95% CI"
        )

        if next_x is not None:
            ax[0].axvline(next_x.item(), linestyle="dotted", c="k")

        ax[0].legend()
        ax[0].set_ylabel("predictive")

        # acquisition score
        ax[1].plot(xs, acquisition_score, c="g")
        ax[1].fill_between(xs.flatten(), acquisition_score, 0, color="g", alpha=0.5)

        if next_x is not None:
            ax[1].axvline(next_x.item(), linestyle="dotted", c="k")

        ax[1].set_ylabel("acquisition score")
        
        plt.show()

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
    # declare the GP
    noise = 1e-4

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.likelihood.noise = noise

    # train the hyperparameter (the constant)
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

# Initialize Forrester function and bounds
forrester = Forrester()
bounds = forrester.bounds  # [0, 1]

# Set up grid for visualization
xs = torch.linspace(bounds[0, 0], bounds[1, 0], 101).unsqueeze(1)
ys = forrester(xs)

# Parameters
n_initial_points = 10
num_queries = 10
num_runs = 10

# Store minimum f(x) for each run
all_min_f_values = []

for run in range(num_runs):
    print(f"\nRun {run + 1}/{num_runs}")
    
    # Set different random seed for each run
    torch.manual_seed(2 + run)
    
    # Initialize with 10 random points in [0, 1]
    train_x = torch.rand(size=(n_initial_points, 1))  # Uniformly sampled from [0, 1]
    train_y = forrester(train_x)

    # Print initial points and their function values
    print("Initial points:")
    print(torch.hstack([train_x, train_y.unsqueeze(1)]))

    # Fit initial GP model
    model, likelihood = fit_gp_model(train_x, train_y)

    # Track minimum f(x) for this run
    min_f_values = [train_y.min().item()]

    # Run Bayesian optimization
    for i in range(num_queries):
        print(f"Iteration {i}, incumbent x={train_x[train_y.argmin()].item():.4f}, f(x)={train_y.min().item():.4f}")

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

        # Comment out to avoid clutter
        # visualize_gp_belief_and_policy(model, likelihood, policy, next_x=next_x)

        next_y = forrester(next_x)

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

# Plot mean ± std (optional, for additional context)
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
plt.title('Convergence Plot: Mean, Std, and 95% CI over 10 Runs')
plt.grid(True)
plt.legend()
plt.savefig('convergence_plot.png')
plt.close()