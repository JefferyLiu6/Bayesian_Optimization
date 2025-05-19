import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.acquisition.analytic import LogExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from functools import partial
from tqdm import tqdm

from functions import Forrester, Branin, Hartmann, Levy
from model import build_gp
from convergence_plot import plot_single_run, plot_convergence

# ---------------------------------------------------------
# config
# ---------------------------------------------------------
device = torch.device("cpu")

POLICIES = {
    "pi": ProbabilityOfImprovement,
    "ei": LogExpectedImprovement,
    "lcb": partial(UpperConfidenceBound, beta=1.0),  # LCB for minimization
}

OBJECTIVES = {
    "forrester": Forrester,
    "branin": Branin,
    "hartmann3": partial(Hartmann, dim=3),
    "hartmann4": partial(Hartmann, dim=4),
    "hartmann6": partial(Hartmann, dim=6),
    "levy10": partial(Levy, dim=10),
}

# ---------------------------------------------------------
# visualization for 1D functions
# ---------------------------------------------------------
def visualize_gp_belief_and_policy(model, likelihood, policy, train_x, train_y, obj, output_dir, iteration, policy_name):
    """Visualize GP belief and acquisition score for 1D functions."""
    if train_x.size(-1) != 1:  # Only for 1D functions
        return

    with torch.no_grad():
        # Generate test points
        bounds = obj.bounds
        xs = torch.linspace(bounds[0, 0], bounds[1, 0], 1000, device=device).unsqueeze(1)
        ys = obj(xs).squeeze()

        # Normalize test points for GP prediction
        lb, ub = bounds[0], bounds[1]
        xs_normalized = (xs - lb) / (ub - lb)

        # Get predictive distribution
        predictive_distribution = likelihood(model(xs_normalized))
        predictive_mean = predictive_distribution.mean
        predictive_upper, predictive_lower = predictive_distribution.confidence_region()

        # Get acquisition scores
        acquisition_score = policy(xs_normalized.unsqueeze(1)).squeeze()

    # Create plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # GP belief
    ax[0].plot(xs.cpu(), ys.cpu(), label="objective", c="r")
    ax[0].scatter(train_x.cpu(), train_y.cpu(), marker="x", c="k", label="observations")
    ax[0].plot(xs.cpu(), predictive_mean.cpu(), label="mean")
    ax[0].fill_between(
        xs.flatten().cpu(), predictive_upper.cpu(), predictive_lower.cpu(), alpha=0.3, label="95% CI"
    )
    ax[0].set_ylabel("f(x)")
    ax[0].legend()

    # Acquisition score
    ax[1].plot(xs.cpu(), acquisition_score.cpu(), c="g")
    ax[1].fill_between(xs.flatten().cpu(), acquisition_score.cpu(), 0, color="g", alpha=0.5)
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Acquisition score")

    plt.suptitle(f"Iteration {iteration + 1} ({policy_name.upper()})")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"gp_belief_iter_{iteration + 1}_{policy_name.lower()}.png"))
    plt.close()

# ---------------------------------------------------------
# main optimisation loop
# ---------------------------------------------------------
def run_one_bo_loop(obj, acq_name, iters, n_init, args):
    d = obj.bounds.size(1)

    # ---------- initial random design ----------
    train_x = (obj.bounds[0] +
               (obj.bounds[1] - obj.bounds[0]) *
               torch.rand(n_init, d)).to(device)
    train_y = obj(train_x).squeeze()  # Ensure train_y is 1D

    min_f_values = [train_y.min().item()]

    AcqClass = POLICIES[acq_name]

    for i in tqdm(range(iters), desc="BO Iteration"):
        # ---------- fit GP ----------
        model, likelihood = build_gp(train_x, train_y, obj.bounds)

        # ---------- acquisition function ----------
        if acq_name == "pi":
            acq = AcqClass(model=model, best_f=train_y.min(), maximize=False)
        elif acq_name == "ei":
            acq = AcqClass(model=model, best_f=train_y.min())
        else:  # lcb
            acq = AcqClass(model=model)

        # ---------- optimize acquisition ----------
        candidate, _ = optimize_acqf(
            acq,
            bounds=(obj.bounds / (obj.bounds[1] - obj.bounds[0])).to(device),  # Normalized bounds
            q=1,
            num_restarts=30,
            raw_samples=100,
        )
        candidate = candidate.view(1, d)  # Ensure shape (1,d)
        candidate = candidate * (obj.bounds[1] - obj.bounds[0]) + obj.bounds[0]  # Denormalize

        # ---------- evaluate objective ----------
        new_y = obj(candidate).squeeze()  # Ensure new_y is scalar or 1D
        new_y = new_y.unsqueeze(0) if new_y.dim() == 0 else new_y

        # ---------- update data ----------
        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        # ---------- visualize for 1D functions ----------
        visualize_gp_belief_and_policy(
            model, likelihood, acq, train_x, train_y, obj,
            os.path.join(args.fn.lower(), args.policy.lower()), i, acq_name
        )

        min_f_values.append(train_y.min().item())

    return min_f_values

def main(args):
    torch.manual_seed(args.seed)

    obj_cls = OBJECTIVES[args.fn.lower()]
    obj = obj_cls()

    out_dir = os.path.join(args.fn.lower(), args.policy.lower())
    os.makedirs(out_dir, exist_ok=True)

    all_min_f = []
    for run in range(args.runs):
        torch.manual_seed(args.seed + run)  # Vary seed per run
        mf = run_one_bo_loop(obj, args.policy, args.iters, args.n_init, args)
        all_min_f.append(mf)
        plot_single_run(mf, run, out_dir, args.policy)
        print(f"✓ finished run {run + 1}/{args.runs}")

    plot_convergence(all_min_f, out_dir, args.policy)
    print(f"\n✅ all plots saved to '{out_dir}/'")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fn",     choices=OBJECTIVES.keys(), default="forrester")
    p.add_argument("--policy", choices=POLICIES.keys(),   default="ei")
    p.add_argument("--iters",  type=int, default=50, help="BO iterations")
    p.add_argument("--n_init", type=int, default=10, help="initial points")
    p.add_argument("--runs",   type=int, default=10, help="repetitions")
    p.add_argument("--seed",   type=int, default=0,  help="torch seed")
    main(p.parse_args())