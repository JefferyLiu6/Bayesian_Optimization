import argparse
import os
import numpy as np
import torch
import warnings
from botorch.acquisition.analytic import LogExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from functools import partial
from tqdm import tqdm

from functions import Forrester, Branin, Hartmann, Levy
from model import build_gp
from convergence_plot import plot_single_run, plot_convergence, plot_function_and_points
from utils import save_points, load_points

# config
device = torch.device("cpu")

POLICIES = {
    "pi": ProbabilityOfImprovement,
    "ei": LogExpectedImprovement,
    "lcb": partial(UpperConfidenceBound, beta=2.0),
}

OBJECTIVES = {
    "forrester": Forrester,
    "branin": Branin,
    "hartmann3": partial(Hartmann, dim=3),
    "hartmann4": partial(Hartmann, dim=4),
    "hartmann6": partial(Hartmann, dim=6),
    "levy10": partial(Levy, dim=10),
}

FIDELITIES = {
    "low": {"n_init": 5, "iters": 10},
    "medium": {"n_init": 10, "iters": 15},
    "high": {"n_init": 15, "iters": 20},
}

def run_one_bo_loop(obj, acq_name, iters, n_init, kernel_type, output_dir, fidelity, run, prev_fidelity=None):
    """Run one BO loop, initializing with all points from previous fidelity if provided."""
    d = obj.bounds.size(1)

    # Initialize with previous fidelity points if provided
    if prev_fidelity:
        try:
            prev_train_x, prev_train_y = load_points(output_dir, prev_fidelity, run)
            # Use all previous points and add new Sobol points to reach n_init if needed
            if prev_train_x.size(0) < n_init:
                extra_n = n_init - prev_train_x.size(0)
                extra_x = draw_sobol_samples(bounds=obj.bounds, n=extra_n, q=1).squeeze(1).to(device)
                extra_y = obj(extra_x).squeeze()
                train_x = torch.cat([prev_train_x, extra_x], dim=0)
                train_y = torch.cat([prev_train_y, extra_y], dim=0)
            else:
                train_x = prev_train_x
                train_y = prev_train_y
        except FileNotFoundError:
            print(f"No points found for {prev_fidelity}, using Sobol initialization")
            train_x = draw_sobol_samples(bounds=obj.bounds, n=n_init, q=1).squeeze(1).to(device)
            train_y = obj(train_x).squeeze()
    else:
        train_x = draw_sobol_samples(bounds=obj.bounds, n=n_init, q=1).squeeze(1).to(device)
        train_y = obj(train_x).squeeze()

    min_f_values = [train_y.min().item()]
    AcqClass = POLICIES[acq_name]

    for i in tqdm(range(iters), desc=f"BO Iteration ({fidelity})"):
        # Fit GP
        model, likelihood = build_gp(train_x, train_y, kernel_type=kernel_type)

        # Acquisition function
        if acq_name == "pi":
            acq = AcqClass(model=model, best_f=train_y.min(), maximize=False)
        elif acq_name == "ei":
            acq = AcqClass(model=model, best_f=train_y.min(), maximize=False)
        else:  # lcb
            beta = 5.0 * (0.1 / 5.0) ** (i / iters)
            acq = UpperConfidenceBound(model=model, beta=beta, maximize=False)

        # Optimize acquisition
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            candidate, acq_val = optimize_acqf(
                acq,
                bounds=obj.bounds.to(device),
                q=1,
                num_restarts=40,
                raw_samples=100,
            )
        candidate = candidate.view(1, d)
        print(f"Iteration {i+1} ({fidelity}): {acq_name.upper()} value = {acq_val.item()}, Candidate = {candidate.tolist()}")

        # Evaluate objective
        new_y = obj(candidate).squeeze()
        new_y = new_y.unsqueeze(0) if new_y.dim() == 0 else new_y

        # Update data
        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        min_f_values.append(train_y.min().item())

    # Save points
    save_points(train_x, train_y, output_dir, fidelity, run)

    return min_f_values, train_x, train_y

def main(args):
    torch.manual_seed(args.seed)

    obj_cls = OBJECTIVES[args.fn.lower()]
    obj = obj_cls()

    out_dir = os.path.join(args.fn.lower(), args.policy.lower(), "multi_fidelity")
    os.makedirs(out_dir, exist_ok=True)

    all_min_f = {"low": [], "medium": [], "high": []}
    for run in range(args.runs):
        torch.manual_seed(args.seed + run)
        print(f"\nRun {run + 1}/{args.runs}")

        # Low fidelity
        mf_low, train_x_low, train_y_low = run_one_bo_loop(
            obj, args.policy, FIDELITIES["low"]["iters"], FIDELITIES["low"]["n_init"],
            args.kernel, out_dir, "low", run
        )
        all_min_f["low"].append(mf_low)
        plot_single_run(mf_low, run, out_dir, f"{args.policy}_low")

        # Medium fidelity, using low fidelity points
        mf_medium, train_x_medium, train_y_medium = run_one_bo_loop(
            obj, args.policy, FIDELITIES["medium"]["iters"], FIDELITIES["medium"]["n_init"],
            args.kernel, out_dir, "medium", run, prev_fidelity="low"
        )
        all_min_f["medium"].append(mf_medium)
        plot_single_run(mf_medium, run, out_dir, f"{args.policy}_medium")

        # High fidelity, using medium fidelity points
        mf_high, train_x_high, train_y_high = run_one_bo_loop(
            obj, args.policy, FIDELITIES["high"]["iters"], FIDELITIES["high"]["n_init"],
            args.kernel, out_dir, "high", run, prev_fidelity="medium"
        )
        all_min_f["high"].append(mf_high)
        plot_single_run(mf_high, run, out_dir, f"{args.policy}_high")

        # Plot function and points for all fidelities
        plot_function_and_points(obj, out_dir, run, args.policy)

        print(f"✓ Finished run {run + 1}/{args.runs}")

    # Plot convergence for each fidelity
    for fidelity in FIDELITIES:
        plot_convergence(all_min_f[fidelity], out_dir, f"{args.policy}_{fidelity}")
    print(f"\nAll plots saved to '{out_dir}/'")

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fn",     choices=OBJECTIVES.keys(), default="forrester")
    p.add_argument("--policy", choices=POLICIES.keys(),   default="ei")
    p.add_argument("--runs",   type=int, default=10, help="repetitions")
    p.add_argument("--seed",   type=int, default=0,  help="torch seed")
    p.add_argument("--kernel", choices=["rbf", "matern"], default="rbf", help="GP kernel type")
    main(p.parse_args())