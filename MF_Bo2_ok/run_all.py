import subprocess

FNS      = ["hartmann3", "hartmann4", "hartmann6", "levy10"]
POLICIES = ["pi", "ei", "lcb"]
COMMON   = ["--iters", "100", "--n_init", "10", "--kernel", "rbf"]

for fn in FNS:
    for policy in POLICIES:
        cmd = ["python", "main.py", "--fn", fn, "--policy", policy] + COMMON
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
