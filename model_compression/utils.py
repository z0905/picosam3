import torch
import torch.nn.functional as F
import numpy as np

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm_for_vis(img_norm: torch.Tensor) -> np.ndarray:
    img = img_norm.detach().cpu()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()

def pad_bbox_to_square(bbox, img_width, img_height, padding=0.1):
    x, y, w, h = bbox
    
    
    pad_w = w * padding
    pad_h = h * padding
    x -= pad_w
    y -= pad_h
    w += 2 * pad_w
    h += 2 * pad_h
    
    
    size = max(w, h)
    cx = x + w / 2
    cy = y + h / 2
    x1 = cx - size / 2
    y1 = cy - size / 2
    x2 = x1 + size
    y2 = y1 + size
    
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    return int(x1), int(y1), int(x2), int(y2)


def steeper_sigmoid(x, temperature=5.0):
    return torch.sigmoid(x * temperature)

def bce_dice_loss(pred_mask, gt_mask):
    pred_mask = steeper_sigmoid(pred_mask)

    bce = F.binary_cross_entropy(pred_mask, gt_mask)
    intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    return bce + dice

def area_loss(pred_mask, gt_mask, min_ratio=0.4):
    pred = steeper_sigmoid(pred_mask)
    pred_area = pred.sum(dim=(2, 3))
    gt_area = gt_mask.sum(dim=(2, 3)) + 1e-6
    ratio = pred_area / gt_area
    return torch.relu(min_ratio - ratio).mean()

def mse_dice_loss(pred_mask, soft_mask):
    pred_mask =steeper_sigmoid(pred_mask)
    soft_mask =steeper_sigmoid(soft_mask)

    mse = F.mse_loss(pred_mask, soft_mask)
    intersection = (pred_mask * soft_mask).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + soft_mask.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    return mse + dice

def compute_iou(pred_mask, target_mask):
    pred_binary = (steeper_sigmoid(pred_mask) > 0.5).detach().cpu().numpy()
    target_binary = (target_mask > 0.5).detach().cpu().numpy()
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    return float(intersection / union) if union > 0 else 1.0