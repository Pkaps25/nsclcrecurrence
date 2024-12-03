import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import subprocess


# Define the different argument values as lists
dilate = [10, 20]
box_size = [25, 35]
lr = [5e-4, 1e-3]
size = ["128"]
# dataset = ["r17", "prasad"]
devices = list(range(4))

# Use itertools.product to generate all combinations of arguments
combinations = list(itertools.product(dilate, box_size, lr))

# Iterate over each combination and spawn the process


def run_command(combo, device):
    # resample = ("32", "48", "48") if combo[0] == "luna" else ("64", "64", "64",)
    # dilate = "3" if combo[0] == "luna" else "10"
    cmd = [
        "python",
        "ct_lung_class/run_single.py",
        "--epochs",
        "2000",
        "--batch-size",
        "16",
        "--dilate",
        str(combo[0]),
        "--resample",
        "128",
        "128",
        "128",
        "--device",
        str(device),
        "--model",
        "densenet",
        "--optimizer",
        "adam",
        "--dataset",
        "sclc",
        "--weight-decay",
        "0.001",
        "--learn-rate",
        str(combo[2]),
        "--tag",
        "middle-slice-2d",
        "--momentum",
        "0.99",
        "--box-size",
        str(combo[1]),
        # "--oversample"
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
