import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from utils import load_points

def plot_single_run(min_f_values, run, output_dir, policy):
    """Plot minimum f(x) vs. iteration for a single run."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(min_f_values)), min_f_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Minimum f(x)')
    plt.title(f'Minimum f(x) vs Iteration (Run {run + 1}, {policy.upper()})')
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'min_f_vs_iteration_run_{run + 1}_{policy.lower()}.png'))
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

def plot_function_and_points(obj, output_dir, run, policy, fidelities=["low", "medium", "high"]):
    """Plot the 1D objective function and points for each fidelity level."""
    # Check if the objective is 1D
    if obj.bounds.size(1) != 1:
        print(f"Cannot plot function for {obj.__class__.__name__} (dimension > 1). Skipping.")
        return

    plt.figure(figsize=(10, 6))

    # Plot the true function (for 1D Forrester)
    x = torch.linspace(obj.bounds[0, 0], obj.bounds[1, 0], 100).view(-1, 1)
    y = obj(x).numpy()
    plt.plot(x.numpy(), y, 'k-', label='Objective Function')

    # Plot points for each fidelity
    colors = {'low': 'ro', 'medium': 'bs', 'high': 'g^'}  # Red circles, blue squares, green triangles
    labels = {'low': 'Low Fidelity', 'medium': 'Medium Fidelity', 'high': 'High Fidelity'}

    for fidelity in fidelities:
        try:
            train_x, train_y = load_points(output_dir, fidelity, run)
            plt.plot(train_x.numpy(), train_y.numpy(), colors[fidelity], label=labels[fidelity], markersize=8)
        except FileNotFoundError:
            print(f"No points found for {fidelity} fidelity, run {run + 1}. Skipping.")

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Objective Function and Sampled Points (Run {run + 1}, {policy.upper()})')
    plt.grid(True)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'function_and_points_run_{run + 1}_{policy.lower()}.png'))
    plt.close()