import torch
import numpy as np

from constants import CMAP


def batchnorm_to_instancenorm(module: torch.nn.Module):
    """
    Replace all batchnorm layers with instancenorm.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            child: torch.nn.BatchNorm2d = child
            setattr(module, name, torch.nn.InstanceNorm2d(child.num_features))
        else:
            batchnorm_to_instancenorm(child)

def batchnorm_to_groupnorm(module: torch.nn.Module, num_groups=1):
    """
    Replace all batchnorm layers with groupnorm. Keeping num_groups=1 is equivalent to layernorm.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            child: torch.nn.BatchNorm2d = child
            setattr(module, name, torch.nn.GroupNorm(num_groups, child.num_features))
        else:
            batchnorm_to_groupnorm(child)

def img_display(img_np):
    # Normalize to 0-255 and convert to uint8 for display purposes
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np -= np.min(img_np)
    img_np *= 255.0 / np.max(img_np)
    img_np = img_np.astype(np.uint8)
    return img_np

def mask_display(tensor):
    # Map mask values to RGB for display purposes
    mask_np = tensor.cpu().detach().numpy()
    disp_array = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for cls_id, cls_color in enumerate(CMAP.values()):
        locs = mask_np == cls_id
        disp_array[locs] = cls_color
    return disp_array

def class_IoUs(confusion_matrix):
    # Calculate per class IoUs given a confusion matrix
    tp = confusion_matrix.diagonal()
    fp = np.sum(confusion_matrix, axis=0) - tp
    fn = np.sum(confusion_matrix, axis=1) - tp
    class_ious = tp / (tp + fp + fn)
    return class_ious
