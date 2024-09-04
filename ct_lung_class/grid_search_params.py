import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Define the different argument values as lists
learn_rate = [0.001, 0.0001]
model = ["monai.networks.nets.densenet169", "monai.networks.nets.densenet121"]
devices = list(range(4))
padding = ["border", "zeros"]
# Use itertools.product to generate all combinations of arguments
combinations = list(itertools.product(learn_rate, model, padding))

# Iterate over each combination and spawn the process


def run_command(combo, device):
    cmd = [
        "python",
        "ct_lung_class/train.py",
        "--epochs",
        "10000",
        "--batch-size",
        "16",
        "--affine-prob",
        "0.6",
        "--translate",
        "10",
        "--scale",
        "0.15",
        "--dilate",
        "10",
        "--resample",
        "64",
        "--val-ratio",
        "0.2",
        "--padding",
        str(combo[2]),
        "--device",
        str(device),
        "--learn-rate",
        str(combo[0]),
        "--momentum",
        "0.99",
        "--weight-decay",
        "0.001",
        "--model",
        str(combo[1]),
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
