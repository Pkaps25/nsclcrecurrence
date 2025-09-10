from .medicalnet import PretrainedMedicalNet
import torch.nn as nn 



def freeze_backbone(model: nn.Module):
    for name, param in model.net.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


__all__ = [PretrainedMedicalNet, freeze_backbone, unfreeze_all]
