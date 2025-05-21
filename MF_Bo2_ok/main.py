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
from convergence_plot import plot_single_run, plot_convergence

# config
device = torch.device("cpu")

POLICIES = {
    "pi": ProbabilityOfImprovement,
    "ei": LogExpectedImprovement,
    "lcb": partial(UpperConfidenceBound, beta=2.0),  # LCB for minimization
}

OBJECTIVES = {
    "forrester": Forrester,
    "branin": Branin,
    "hartmann3": partial(Hartmann, dim=3),
    "hartmann4": partial(Hartmann, dim=4),
    "hartmann6": partial(Hartmann, dim=6),
    "levy10": partial(Levy, dim=10),
}


# main optimisation loop
def run_one_bo_loop(obj, acq_name, iters, n_init, kernel_type):
    d = obj.bounds.size(1)

    #  initial Sobol design
    train_x = draw_sobol_samples(bounds=obj.bounds, n=n_init, q=1).squeeze(1).to(device)
    train_y = obj(train_x).squeeze()  # Ensure train_y is 1D

    min_f_values = [train_y.min().item()]

    AcqClass = POLICIES[acq_name]

    for i in tqdm(range(iters), desc="BO Iteration"):
        # fit GP
        model, likelihood = build_gp(train_x, train_y, kernel_type=kernel_type)

        # acquisition function
        if acq_name == "pi":
            acq = AcqClass(model=model, best_f=train_y.min(), maximize=False)
        elif acq_name == "ei":
            acq = AcqClass(model=model, best_f=train_y.min(), maximize=False)
        else:  # lcb
            beta = 5.0 * (0.1 / 5.0) ** (i / iters)  # Dynamic beta for LCB
            acq = UpperConfidenceBound(model=model, beta=beta, maximize=False)

        #  optimise acquisition
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            candidate, acq_val = optimize_acqf(
                acq,
                bounds=obj.bounds.to(device),
                q=1,
                num_restarts=40,
                raw_samples=100,
            )
        candidate = candidate.view(1, d)  # ensure shape (1,d)
        print(f"Iteration {i+1}: {acq_name.upper()} value = {acq_val.item()}, Candidate = {candidate.tolist()}")

        #  evaluate objective 
        new_y = obj(candidate).squeeze()  # Ensure new_y is scalar or 1D
        new_y = new_y.unsqueeze(0) if new_y.dim() == 0 else new_y

        #  update data 
        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

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
        mf = run_one_bo_loop(obj, args.policy, args.iters, args.n_init, args.kernel)
        all_min_f.append(mf)
        plot_single_run(mf, run, out_dir, args.policy)
        print(f"✓ finished run {run + 1}/{args.runs}")

    plot_convergence(all_min_f, out_dir, args.policy)
    print(f"\n all plots saved to '{out_dir}/'")

# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fn",     choices=OBJECTIVES.keys(), default="forrester")
    p.add_argument("--policy", choices=POLICIES.keys(),   default="ei")
    p.add_argument("--iters",  type=int, default=30, help="BO iterations")
    p.add_argument("--n_init", type=int, default=20, help="initial points")
    p.add_argument("--runs",   type=int, default=10, help="repetitions")
    p.add_argument("--seed",   type=int, default=0,  help="torch seed")
    p.add_argument("--kernel", choices=["rbf", "matern"], default="rbf", help="GP kernel type")
    main(p.parse_args())