import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Define the different argument values as lists
learn_rates = [0.001, 0.0001]
boxes = [65]
weight_decay = [0.001]
devices = list(range(4))
# Use itertools.product to generate all combinations of arguments
combinations = list(itertools.product(learn_rates, boxes, weight_decay))

# Iterate over each combination and spawn the process


def run_command(combo, device):
    cmd = [
        "python",
        "ct_lung_class/run_single.py",
        "--epochs",
        "20000",
        "--batch-size",
        "16",
        "--dilate",
        "15",
        "--resample",
        str(combo[1]),  str(combo[1]), str(combo[1]),
        "--box-size",
        str(combo[1]),  str(combo[1]), str(combo[1]),
        "--val-ratio",
        "0.15",
        "--test-ratio",
        "0.15",
        "--dropout",
        "0.3",
        "--learn-rate",
        str(combo[0]),
        "--momentum",
        "0.99",
        "--weight-decay",
        str(combo[2]),
        "--model",
        "densenet",
        "--optimizer",
        "adam",
        "--tag",
        f"grid-search-lr-{str(combo[0])}-size-{str(combo[1])}-l2-{str(combo[2])}--zara-dn201",
        "--device",
        str(device),
        "--dataset",
        "zara",
        "--fixed-size"
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
