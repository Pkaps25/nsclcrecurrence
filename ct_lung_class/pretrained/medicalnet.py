import logging
from typing import Any, Sequence

import torch
from monai.networks.nets import ResNet, resnet50


import torch
import logging
from typing import Any, Sequence, Callable
from monai.networks.nets import resnet50, ResNet
import torch.nn as nn


class PretrainedMedicalNet(nn.Module):
    """
    A MONAI ResNet wrapper that loads MedicalNet pretrained weights.

    References:
    - https://github.com/Project-MONAI/MONAI
    - https://github.com/Borda/MedicalNet
    """

    def __init__(
        self,
        pretrained_path: str,
        model_constructor: Callable = resnet50,
        spatial_dims: int = 3,
        n_input_channels: int = 1,
        num_classes: int = 2,
        **kwargs_monai_resnet: Any,
    ) -> None:
        super().__init__()
        self.pretrained_path = pretrained_path

        self.net: ResNet = model_constructor(
            pretrained=False,
            spatial_dims=spatial_dims,
            n_input_channels=n_input_channels,
            num_classes=num_classes,
            **kwargs_monai_resnet,
        )

        self.inside_layers = self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> Sequence[str]:
        net_dict = self.net.state_dict()
        pretrain = torch.load(self.pretrained_path, weights_only=False)

        # Remove "module." prefix if present
        pretrain["state_dict"] = {
            k.replace("module.", ""): v for k, v in pretrain["state_dict"].items()
        }

        missing = tuple(k for k in net_dict.keys() if k not in pretrain["state_dict"])
        logging.debug(f"missing in pretrained: {len(missing)}")

        inside = tuple(k for k in pretrain["state_dict"] if k in net_dict.keys())
        logging.debug(f"inside pretrained: {len(inside)}")

        unused = tuple(k for k in pretrain["state_dict"] if k not in net_dict.keys())
        logging.debug(f"unused pretrained: {len(unused)}")

        assert len(inside) > len(missing)
        assert len(inside) > len(unused)

        # Keep only matching keys
        filtered_state_dict = {
            k: v for k, v in pretrain["state_dict"].items() if k in net_dict
        }

        self.net.load_state_dict(filtered_state_dict, strict=False)
        return inside

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_loaded_layers(self) -> Sequence[str]:
        """Return the names of successfully loaded pretrained layers."""
        return self.inside_layers

