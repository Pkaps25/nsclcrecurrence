import matplotlib.pyplot as plt
import numpy as np
import torch

train_metrics = torch.tensor(torch.load("train_metrics.pth"))
val_metrics = torch.tensor(torch.load("val_metrics.pth"))
window_size = 10

def moving_average(data, window_size):
    """Compute the moving average of a 1D array."""
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

fig, ax = plt.subplots(nrows=3, figsize=(16,8))
fig.tight_layout()


for i, metric in enumerate(("f1", "accuracy", "loss")):
    ax[i].set_title(metric)
    ax[i].plot(moving_average(train_metrics[:,i], window_size), label="train")
    ax[i].plot(moving_average(val_metrics[:,i], window_size), label="validation")

handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.savefig("metrics.png")