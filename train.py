import yaml
from pathlib import Path
import argparse
import random

import torch
from torch.cuda import amp
import wandb
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix

from dataset import Comma10kDataset
from constants import CMAP, MOVEABLE_IDX
import utils
from schedulers import PolynomialLRDecay


# Set random seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(cfg):
    # Get wandb setup
    run = wandb.init(project="comma10k-segnet", config=cfg, entity="methier")
    
    # Create directory for saving checkpoints
    checkpoint_dir = Path(cfg['checkpoint_dir']) / run.name
    checkpoint_dir.mkdir(parents=True)
    
    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimizes training speed
    torch.backends.cudnn.benchmark = True
    
    print("Setting up datasets and data loaders...")
    # Original size is (874, 1164)
    # New size is downscaled and raised to nearest multiple of min_factor (needed for DeeplabV3+)
    img_size = (874, 1164)
    height = int(img_size[0] / cfg["downscale_factor"])
    height = height + (cfg["min_factor"] - height % cfg["min_factor"])
    width = int(img_size[1] / cfg["downscale_factor"])
    width = width + (cfg["min_factor"] - width % cfg["min_factor"])

    train_transforms = A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=50.0, p=0.4),
        A.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.6, hue=0.05, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 13), p=1.0),
            A.Sharpen(alpha=(0.2, 0.6), lightness=(0.5, 0.9), p=1.0),
        ], p=0.4),
        A.CLAHE(p=0.3),
        A.ISONoise(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_set = Comma10kDataset(cfg['root'], 'train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, 
                              batch_size=cfg['train_batch_size'], 
                              shuffle=True, 
                              num_workers=cfg['train_num_workers'],
                              pin_memory=True)
    val_set = Comma10kDataset(cfg['root'], 'val', val_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, 
                              batch_size=cfg['val_batch_size'], 
                              shuffle=True, 
                              num_workers=cfg['val_num_workers'],
                              pin_memory=True)
    
    print("Initializing model and loss...")
    # Init model
    model = getattr(smp, cfg["model"])(encoder_name=cfg["encoder"], classes=len(CMAP))

    # Replace all batch norm with instance norm since we are using very small batch sizes
    #utils.batchnorm_to_instancenorm(model)

    model.to(device)

    # Init loss and optimizer
    ce_loss = torch.nn.CrossEntropyLoss()
    focal_loss = smp.losses.FocalLoss('multiclass')
    #dice_loss = smp.losses.DiceLoss('multiclass', classes=len(CMAP), log_loss=True)

    if cfg["opt_name"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], momentum=0.9)
    elif cfg["opt_name"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if cfg["scheduler"] == "poly":
        # Use poly lr scheduler as suggested in DeepLabV3 paper
        scheduler = PolynomialLRDecay(optimizer, len(train_loader) * cfg["epochs"], end_learning_rate=cfg["lr"]/100, power=0.9)
    elif cfg["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    # Create a GradScaler once at the beginning of training
    scaler = amp.GradScaler()

    # For tracking best model through run
    best_iou = float("-inf")

    for epoch in range(cfg['epochs']):
        # Training loop
        print(f"\n=== Epoch {epoch + 1} ===")
        log_dict = {
            "train": {
                "focal": 0.0, "ce": 0.0, "dice": 0.0, "cm": np.zeros((len(CMAP), len(CMAP)))
            },
            "val": {
                "focal": 0.0, "ce": 0.0, "dice": 0.0, "cm": np.zeros((len(CMAP), len(CMAP)))
            }
        }
        model.train()
        for idx, sample in enumerate(tqdm(train_loader, desc='Training')):
            optimizer.zero_grad()
            inputs = sample["img"].to(device)
            labels = sample["mask"].to(device)

            # Run the forward pass with autocasting
            with amp.autocast():
                output = model(inputs)
                focal = focal_loss(output, labels)
                ce = ce_loss(output, labels)

            log_dict["train"]["focal"] += focal.item()
            log_dict["train"]["ce"] += ce.item()

            # Collect confusion matrix to calculate IoU
            preds = torch.argmax(output, axis=1)
            conf_matrix = confusion_matrix(
                labels.cpu().detach().numpy().ravel(),
                preds.cpu().detach().numpy().ravel(),
                labels=list(range(len(CMAP)))
            )
            log_dict["train"]["cm"] += conf_matrix

            # Backward and optimize
            scaler.scale(focal).backward()
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            
            if cfg["scheduler"] == "poly":
                scheduler.step()

            # Grab last examples in epoch for logging
            if idx == len(train_loader) - 1:
                log_dict["train"]["img"] = utils.img_display(inputs[0].cpu().detach().numpy())
                log_dict["train"]["mask"] = labels[0].cpu().detach().numpy()
                log_dict["train"]["pred"] = torch.argmax(output[0], axis=0).cpu().detach().numpy()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(val_loader, desc='Validation')):
                inputs = sample["img"].to(device)
                labels = sample["mask"].to(device)

                # Forward pass
                output = model(inputs)
                loss = focal_loss(output, labels)
                ce = ce_loss(output, labels)

                log_dict["val"]["focal"] += loss.item()
                log_dict["val"]["ce"] += ce.item()

                # Collect confusion matrix to calculate IoU
                preds = torch.argmax(output, axis=1)
                conf_matrix = confusion_matrix(
                    labels.cpu().detach().numpy().ravel(),
                    preds.cpu().detach().numpy().ravel(),
                    labels=list(range(len(CMAP)))
                )
                log_dict["val"]["cm"] += conf_matrix

                # Grab last examples in epoch for logging
                if idx == len(val_loader) - 1:
                    log_dict["val"]["img"] = utils.img_display(inputs[0].cpu().detach().numpy())
                    log_dict["val"]["mask"] = labels[0].cpu().detach().numpy()
                    log_dict["val"]["pred"] = torch.argmax(output[0], axis=0).cpu().detach().numpy()

        # Prep seg images for logging
        wandb_cmap = dict(zip(range(len(CMAP)), CMAP.keys()))
        train_log_img = wandb.Image(log_dict["train"]["img"], masks={
            "predictions": {
                "mask_data": log_dict["train"]["pred"],
                "class_labels": wandb_cmap
            },
            "ground_truth": {
                "mask_data": log_dict["train"]["mask"],
                "class_labels": wandb_cmap
            }
        })
        val_log_img = wandb.Image(log_dict["val"]["img"], masks={
            "predictions": {
                "mask_data": log_dict["val"]["pred"],
                "class_labels": wandb_cmap
            },
            "ground_truth": {
                "mask_data": log_dict["val"]["mask"],
                "class_labels": wandb_cmap
            }
        })

        # We want to track moveable class IoU and mIoU
        train_ious = utils.class_IoUs(log_dict["train"]["cm"])
        val_ious = utils.class_IoUs(log_dict["val"]["cm"])

        # Save checkpoint for best epoch
        val_miou = np.mean(val_ious)
        if val_miou > best_iou:
            best_iou = val_miou
            wandb.run.summary["best_iou"] = best_iou
            torch.save(model.state_dict(), checkpoint_dir / f"{run.name}_best_model.pt")

        current_lr = optimizer.param_groups[0]['lr'] if cfg["scheduler"] == "plateau" else scheduler.get_lr()[0]
        wandb.log({
            'train_ce_loss': log_dict["train"]["ce"] / len(train_loader),
            'train_focal_loss': log_dict["train"]["focal"] / len(train_loader),
            'val_ce_loss': log_dict["val"]["ce"] / len(val_loader),
            'val_focal_loss': log_dict["val"]["focal"] / len(val_loader),
            'train_imgs': train_log_img,
            'val_imgs': val_log_img,
            'train_mIoU': np.mean(train_ious),
            'train_moveable_IoU': train_ious[MOVEABLE_IDX],
            'val_mIoU': val_miou,
            'val_moveable_IoU': val_ious[MOVEABLE_IDX],
            'lr': current_lr,
        })

        if cfg["scheduler"] == "plateau":
            scheduler.step(log_dict["val"]["ce"] / len(val_loader))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_name', type=str, help='Name of the config file located in configs')
    args = parser.parse_args()
    
    # Load in config
    cfg_path = Path(__file__).parent.absolute() / args.cfg_name
    with cfg_path.open('r') as f:
        cfg = yaml.full_load(f)
    print(cfg)
    
    main(cfg)
