# convergence_plot.py
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from botorch.models.model import Model
from gpytorch.likelihoods import GaussianLikelihood
from functions import Forrester  # Import Forrester class


def plot_single_run(min_f_values, run, iteration, output_dir, policy):
    """Plot minimum f(x) vs. iteration for a single run and specific iteration."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(min_f_values)), min_f_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Minimum f(x)')
    plt.title(f'Minimum f(x) vs Iteration (Run {run + 1}, {policy.upper()}, Iteration {iteration + 1})')
    plt.grid(True)
    # Create directory structure: multi_fidelity/forrester/<policy>/run<run_number>/iter<iteration_number>
    run_dir = os.path.join(output_dir, f'run{run + 1}')
    iter_dir = os.path.join(run_dir, f'iter{iteration + 1}')
    os.makedirs(iter_dir, exist_ok=True)
    plt.savefig(os.path.join(iter_dir, f'min_f_vs_iteration_run_{run + 1}_iter_{iteration + 1}_{policy.lower()}.png'))
    plt.close()


def plot_convergence(all_min_f_values, output_dir, policy):
    """Plot mean, std, and 95% CI of min f(x) across multiple runs."""
    all_min_f_values = np.array(all_min_f_values)
    num_runs, num_iters = all_min_f_values.shape
    mean_f = np.mean(all_min_f_values, axis=0)
    std_f = np.std(all_min_f_values, axis=0)
    ci_95 = 1.96 * std_f / np.sqrt(num_runs)

    iterations = range(num_iters)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_f, marker='o', linestyle='-', color='b', label='Mean Minimum f(x)')
    plt.fill_between(iterations, mean_f - ci_95, mean_f + ci_95, color='b', alpha=0.2, label='95% Confidence Interval')
    plt.fill_between(iterations, mean_f - std_f, mean_f + std_f, color='g', alpha=0.1, label='Mean ± Std')
    plt.xlabel('Iteration')
    plt.ylabel('Minimum f(x)')
    plt.title(f'Convergence Plot: Mean, Std, and 95% CI over Runs ({policy.upper()})')
    plt.grid(True)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'convergence_plot_{policy.lower()}.png'))
    plt.close()


def plot_fidelity_function(obj, train_x, train_y, model: Model, likelihood: GaussianLikelihood, fidelity, output_dir, run, iteration):
    """Plot the Forrester function, observed points, and GP predictions for a given fidelity and iteration."""
    plt.figure(figsize=(10, 6))
    
    # Generate points for plotting the true function
    x_plot = torch.linspace(0, 1, 200).reshape(-1, 1).to(train_x)
    with torch.no_grad():
        # True function
        y_true = obj(x_plot).cpu().numpy()
        # GP predictions
        model.eval()
        likelihood.eval()
        posterior = model.posterior(x_plot)
        mean = posterior.mean.cpu().numpy().squeeze()
        std = torch.sqrt(posterior.variance).cpu().numpy().squeeze()
    
    # Plot true function
    plt.plot(x_plot.cpu().numpy(), y_true, 'k-', label='True Function')
    # Plot GP mean and uncertainty
    plt.plot(x_plot.cpu().numpy(), mean, 'b-', label='GP Mean')
    plt.fill_between(
        x_plot.cpu().numpy().squeeze(),
        mean - 1.96 * std,
        mean + 1.96 * std,
        color='b',
        alpha=0.2,
        label='95% CI'
    )
    # Plot observed points
    plt.scatter(
        train_x.cpu().numpy(),
        train_y.cpu().numpy(),
        c='red',
        marker='o',
        s=100,
        label='Observed Points'
    )
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Forrester Function with GP Fit (Run {run + 1}, {fidelity.capitalize()} Fidelity, Iteration {iteration + 1})')
    plt.grid(True)
    plt.legend()
    
    # Create directory structure: multi_fidelity/forrester/<policy>/run<run_number>/iter<iteration_number>
    run_dir = os.path.join(output_dir, f'run{run + 1}')
    iter_dir = os.path.join(run_dir, f'iter{iteration + 1}')
    os.makedirs(iter_dir, exist_ok=True)
    plt.savefig(os.path.join(iter_dir, f'forrester_fit_run_{run + 1}_iter_{iteration + 1}_{fidelity.lower()}.png'))
    plt.close()