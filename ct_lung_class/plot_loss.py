import json

import matplotlib.pyplot as plt

with open("loss.json", "r") as jsonfile:
    loss_dict = json.load(jsonfile)

plt.plot(loss_dict["train"])
plt.plot(loss_dict["val"])
plt.savefig("loss.png")
