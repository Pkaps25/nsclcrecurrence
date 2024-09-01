import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Define the different argument values as lists
learn_rate = [0.001, 0.0001]
momentum = [0.90, 0.99]
l2 = [1e-4, 1e-3]
model = ["monai.networks.nets.densenet121", "monai.networks.nets.densenet169"]
devices = list(range(4))

# Use itertools.product to generate all combinations of arguments
combinations = list(itertools.product(learn_rate, momentum, l2, model))

# Iterate over each combination and spawn the process


def run_command(combo, device):
    cmd = [
        "python",
        "ct_lung_class/train.py",
        "--epochs",
        "3000",
        "--batch-size",
        "16",
        "--affine-prob",
        "0.6",
        "--translate",
        "10",
        "--scale",
        "0.12",
        "--dilate",
        "10",
        "--resample",
        "64",
        "--val-ratio",
        "0.2",
        "--padding",
        "border",
        "--device",
        str(device),
        "--learn-rate",
        str(combo[0]),
        "--momentum",
        str(combo[1]),
        "--weight-decay",
        str(combo[2]),
        "--model",
        str(combo[3]),
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


with ProcessPoolExecutor(max_workers=8) as executor:
    futures = []
    for i, combo in enumerate(combinations):
        device = devices[i % len(devices)]
        futures.append(executor.submit(run_command, combo, device))

    for future in tqdm(as_completed(futures), total=len(combinations)):
        future.result()
