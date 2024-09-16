import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Define the different argument values as lists
batch_size = [2, 4]
model = "monai.networks.nets.densenet121"
decay = [1e-4]
devices = list(range(4))
# Use itertools.product to generate all combinations of arguments
combinations = list(itertools.product(batch_size, decay))

# Iterate over each combination and spawn the process


def run_command(combo, device):
    cmd = [
        "python",
        "ct_lung_class/run.py",
        "--epochs",
        "3000",
        "--batch-size",
        str(combo[0]),
        "--affine-prob",
        "0.75",
        "--translate",
        "15",
        "--scale",
        "0.1",
        "--dilate",
        "10",
        "--resample",
        "64",
        "--val-ratio",
        "0.2",
        "--padding",
        "border",
        "--learn-rate",
        "0.001",
        "--momentum",
        "0.99",
        "--weight-decay",
        str(combo[1]),
        "--model",
        model,
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


with ProcessPoolExecutor(max_workers=4) as executor:
    futures = []
    for i, combo in enumerate(combinations):
        device = devices[i % len(devices)]
        futures.append(executor.submit(run_command, combo, device))

    for future in tqdm(as_completed(futures), total=len(combinations)):
        future.result()
