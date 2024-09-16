import logging
from typing import Any, Sequence, Tuple

import torch
from monai.networks.nets import ResNet, resnet50


def create_pretrained_medical_resnet(
    pretrained_path: str,
    model_constructor: callable = resnet50,
    spatial_dims: int = 3,
    n_input_channels: int = 1,
    num_classes: int = 2,
    **kwargs_monai_resnet: Any,
) -> Tuple[ResNet, Sequence[str]]:
    """This is a specific constructor for MONAI ResNet module loading MedicalNET weights.

    See:
    - https://github.com/Project-MONAI/MONAI
    - https://github.com/Borda/MedicalNet
    """
    net = model_constructor(
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet,
    )
    net_dict = net.state_dict()
    pretrain = torch.load(pretrained_path)
    pretrain["state_dict"] = {
        k.replace("module.", ""): v for k, v in pretrain["state_dict"].items()
    }
    missing = tuple({k for k in net_dict.keys() if k not in pretrain["state_dict"]})
    logging.debug(f"missing in pretrained: {len(missing)}")
    inside = tuple({k for k in pretrain["state_dict"] if k in net_dict.keys()})
    logging.debug(f"inside pretrained: {len(inside)}")
    unused = tuple({k for k in pretrain["state_dict"] if k not in net_dict.keys()})
    logging.debug(f"unused pretrained: {len(unused)}")
    assert len(inside) > len(missing)
    assert len(inside) > len(unused)

    pretrain["state_dict"] = {
        k: v for k, v in pretrain["state_dict"].items() if k in net_dict.keys()
    }
    net.load_state_dict(pretrain["state_dict"], strict=False)
    return net, inside
