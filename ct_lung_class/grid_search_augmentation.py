import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Define the different argument values as lists
epochs = [2000]
batch_size = [16]
affine_prob = [0.75]  # Example of varying affine-prob
translate = [12]  # Example of varying translate
scale = [0.12]
# dilate = [3, 6, 10]
dilate = [6]
resample = [64]
val_ratio = [0.2]
padding = ["border"]

devices = list(range(4))

# Use itertools.product to generate all combinations of arguments
combinations = list(
    itertools.product(
        epochs, batch_size, affine_prob, translate, scale, dilate, resample, val_ratio, padding
    )
)

# Iterate over each combination and spawn the process


def run_command(combo, device):
    cmd = [
        "python",
        "ct_lung_class/train.py",
        "--epochs",
        str(combo[0]),
        "--batch-size",
        str(combo[1]),
        "--affine-prob",
        str(combo[2]),
        "--translate",
        str(combo[3]),
        "--scale",
        str(combo[4]),
        "--dilate",
        str(combo[5]),
        "--resample",
        str(combo[6]),
        "--val-ratio",
        str(combo[7]),
        "--padding",
        str(combo[8]),
        "--device",
        str(device),
        "--model",
        "monai.networks.nets.densenet121",
    ]

    print(f"Running command: {' '.join(cmd)}")
    # subprocess.run(cmd)


with ProcessPoolExecutor(max_workers=8) as executor:
    futures = []
    for i, combo in enumerate(combinations):
        device = devices[i % len(devices)]
        futures.append(executor.submit(run_command, combo, device))

    for future in tqdm(as_completed(futures), total=len(combinations)):
        future.result()
