import os
import numpy as np
import torch

def save_points(train_x, train_y, output_dir, fidelity, run):
    """Save training points to a file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'points_{fidelity}_run_{run + 1}.npz')
    np.savez(file_path, train_x=train_x.cpu().numpy(), train_y=train_y.cpu().numpy())
    print(f"Saved points to {file_path}")

def load_points(output_dir, fidelity, run):
    """Load training points from a file."""
    file_path = os.path.join(output_dir, f'points_{fidelity}_run_{run + 1}.npz')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No points found at {file_path}")
    data = np.load(file_path)
    train_x = torch.tensor(data['train_x'], dtype=torch.float64)
    train_y = torch.tensor(data['train_y'], dtype=torch.float64)
    print(f"Loaded points from {file_path}")
    return train_x, train_y