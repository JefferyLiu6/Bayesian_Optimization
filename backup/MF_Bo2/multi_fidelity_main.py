import argparse
import os
import json
import numpy as np
import torch
import warnings
from botorch.acquisition.analytic import LogExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from functools import partial
from tqdm import tqdm


from functions import Forrester
from model import build_gp
from convergence_plot import plot_single_run, plot_convergence, plot_fidelity_function

# Config
device = torch.device("cpu")

POLICIES = {
    "pi": ProbabilityOfImprovement,
    "ei": LogExpectedImprovement,
    "lcb": partial(UpperConfidenceBound, beta=2.0),  # LCB for minimization
}

FIDELITIES = {
    "low": {"n_init": 3, "iters": 10},
    "medium": {"n_init": 10, "iters": 10},
    "high": {"n_init": 30, "iters": 10},
}

def save_points(data, filename):
    """Save points to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_points(filename):
    """Load points from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def run_one_bo_loop(obj, acq_name, iters, n_init, kernel_type, train_x=None, train_y=None):
    """Run a single BO loop, optionally starting with provided points."""
    d = obj.bounds.size(1)

    # Initialize or use provided points
    if train_x is None or train_y is None:
        train_x = draw_sobol_samples(bounds=obj.bounds, n=n_init, q=1).squeeze(1).to(device)
        train_y = obj(train_x).squeeze()  # Ensure train_y is 1D
    else:
        train_x = train_x.to(device)
        train_y = train_y.to(device).squeeze()

    min_f_values = [train_y.min().item()]
    AcqClass = POLICIES[acq_name]

    for i in tqdm(range(iters), desc="BO Iteration"):
        # Fit GP
        model, likelihood = build_gp(train_x, train_y, kernel_type=kernel_type)

        # Acquisition function
        if acq_name == "pi":
            acq = AcqClass(model=model, best_f=train_y.min(), maximize=False)
        elif acq_name == "ei":
            acq = AcqClass(model=model, best_f=train_y.min(), maximize=False)
        else:  # lcb
            beta = 5.0 * (0.1 / 5.0) ** (i / iters)  # Dynamic beta for LCB
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
        candidate = candidate.view(1, d)  # Ensure shape (1,d)
        print(f"Iteration {i+1}: {acq_name.upper()} value = {acq_val.item()}, Candidate = {candidate.tolist()}")

        # Evaluate objective
        new_y = obj(candidate).squeeze()  # Ensure new_y is scalar or 1D
        new_y = new_y.unsqueeze(0) if new_y.dim() == 0 else new_y

        # Update data
        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        min_f_values.append(train_y.min().item())

    return min_f_values, train_x, train_y, model, likelihood

def run_multi_fidelity_bo(args):
    """Run multi-fidelity BO with point reuse across fidelities."""
    torch.manual_seed(args.seed)
    out_dir = os.path.join("multi_fidelity", args.fn.lower(), args.policy.lower())
    os.makedirs(out_dir, exist_ok=True)

    # Point storage
    points_file = os.path.join(out_dir, "points.json")
    stored_points = load_points(points_file)
    all_min_f = {fidelity: [] for fidelity in FIDELITIES}
    all_points = {fidelity: {"x": [], "y": []} for fidelity in FIDELITIES}

    for run in range(args.runs):
        torch.manual_seed(args.seed + run)
        print(f"\nRun {run + 1}/{args.runs}")
        
        prev_x, prev_y = None, None
        for fidelity in ["low", "medium", "high"]:
            print(f"  Fidelity: {fidelity}")
            obj = Forrester(fidelity=fidelity)
            config = FIDELITIES[fidelity]
            n_init = config["n_init"]
            iters = config["iters"]

            # Reuse points from previous fidelity
            if fidelity == "medium" and prev_x is not None:
                # Start with low fidelity points, re-evaluate
                train_x = prev_x[:5].clone()  # Use first 5 points from low
                train_y = obj(train_x).squeeze()
                # Add additional points
                extra_x = draw_sobol_samples(bounds=obj.bounds, n=n_init-5, q=1).squeeze(1).to(device)
                extra_y = obj(extra_x).squeeze()
                train_x = torch.cat([train_x, extra_x], dim=0)
                train_y = torch.cat([train_y, extra_y], dim=0)
            elif fidelity == "high" and prev_x is not None:
                # Start with medium fidelity points, re-evaluate
                train_x = prev_x[:10].clone()  # Use first 10 points from medium
                train_y = obj(train_x).squeeze()
                # Add additional points
                extra_x = draw_sobol_samples(bounds=obj.bounds, n=n_init-10, q=1).squeeze(1).to(device)
                extra_y = obj(extra_x).squeeze()
                train_x = torch.cat([train_x, extra_x], dim=0)
                train_y = torch.cat([train_y, extra_y], dim=0)
            else:
                # Low fidelity: start fresh
                train_x, train_y = None, None

            # Run BO
            min_f_values, train_x, train_y, model, likelihood = run_one_bo_loop(
                obj, args.policy, iters, n_init, args.kernel, train_x, train_y
            )
            all_min_f[fidelity].append(min_f_values)
            all_points[fidelity]["x"].append(train_x.tolist())
            all_points[fidelity]["y"].append(train_y.tolist())

            # Plot function and points
            plot_fidelity_function(obj, train_x, train_y, model, likelihood, fidelity, out_dir, run)
            plot_single_run(min_f_values, run, out_dir, f"{args.policy}_{fidelity}")

            # Update previous points
            prev_x, prev_y = train_x, train_y

            print(f"  ✓ Finished {fidelity} fidelity")

        # Save points
        stored_points[f"run_{run+1}"] = {
            fid: {"x": all_points[fid]["x"][-1], "y": all_points[fid]["y"][-1]}
            for fid in FIDELITIES
        }
        save_points(stored_points, points_file)

    # Plot convergence for each fidelity
    for fidelity in FIDELITIES:
        plot_convergence(all_min_f[fidelity], out_dir, f"{args.policy}_{fidelity}")

    print(f"\nAll plots and points saved to '{out_dir}/'")

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fn",     choices=["forrester"], default="forrester")
    p.add_argument("--policy", choices=POLICIES.keys(), default="ei")
    p.add_argument("--runs",   type=int, default=10, help="repetitions")
    p.add_argument("--seed",   type=int, default=0, help="torch seed")
    p.add_argument("--kernel", choices=["rbf", "matern"], default="rbf", help="GP kernel type")
    run_multi_fidelity_bo(p.parse_args())