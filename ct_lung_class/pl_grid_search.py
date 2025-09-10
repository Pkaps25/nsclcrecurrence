import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import subprocess

# Available GPUs
devices = [0, 1, 2, 3]
num_devices = len(devices)

# Parameter grids
lrs = [1e-4]
dropouts = [0.2]
weight_decays = [1e-3, 1e-4]
models = ["densenet121"]
losses = ["crossentropy", "focal"]

batches = [16]
max_parallel = 8  # number of concurrent jobs

# Build all parameter combinations
param_grid = list(itertools.product(losses, lrs, dropouts, weight_decays, models, batches))


def run_job(args):
    loss, lr, dp, wd, model, bs, device = args

    tag = f"zara-grid-search-adamw-cosinelr-25k"
    print(f"Launching on GPU {device}: {tag}")

    cmd = [
        "python", "ct_lung_class/pl_main.py",
        "--batch-size", str(bs),
        "--dataset", "zara",
        "--val-ratio", "0.15",
        "--test-ratio", "0.15",
        "--box-size", "65", "65", "65",
        "--fixed-size",
        "--k-folds", "1",
        "--learn-rate", str(lr),
        "--dropout", str(dp),
        "--weight-decay", str(wd),
        "--max-epochs", "22000",
        "--devices", str(device),
        "--tag", tag,
        "--loss", loss,
        "--model", model,
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    futures = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, (loss, lr, dp, wd, model, bs) in enumerate(param_grid):
            device = devices[i % num_devices]
            futures.append(executor.submit(run_job, (loss, lr, dp, wd, model, bs, device)))

        for future in tqdm(as_completed(futures), total=len(param_grid)):
            future.result()

