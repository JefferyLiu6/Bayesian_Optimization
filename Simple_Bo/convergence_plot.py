# convergence_plot.py
import matplotlib.pyplot as plt
import numpy as np
import os


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