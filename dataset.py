from pathlib import Path

import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from constants import CMAP
import utils


class Comma10kDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, root, split, transforms=None):
        self.split = split
        self.root_path = Path(root)
        self.transforms = transforms

        # Load dataset list
        list_file = self.root_path / "files_trainable"
        with list_file.open('r') as f:
            filenames = f.read().splitlines()

        # Split list into train/val where val is any file that ends with 9.png
        val_set = []
        train_set = []
        for fname in filenames:
            if fname[-5] == "9":
                val_set.append(fname)
            else:
                train_set.append(fname)
        if split == "train":
            self.dataset = train_set
        elif split == "val":
            self.dataset = val_set
        else:
            raise ValueError("Invalid split given [train or val]")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load in img and mask (some images randomly have alpha channels, remove it for those cases)
        mask_path = self.root_path / self.dataset[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask)[:, :, :3]
        img_path = self.root_path / "imgs" / mask_path.name
        img = Image.open(img_path)
        img = np.array(img)[:, :, :3]

        # Convert mask from RGB to values from 0 to num_classes-1
        height, width = mask.shape[:2]
        seg_mask = np.zeros((height, width), dtype=np.uint8)
        for cls_id, cls_color in enumerate(CMAP.values()):
            locs = np.all(mask == cls_color, axis=-1)
            seg_mask[locs] = cls_id

        # Apply data augmentation
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=seg_mask)
            img = transformed['image']
            seg_mask = transformed['mask']

            # Scale input img to 0 to 1 and set types to work with CE loss
            img = img.type(torch.float32) / 255.0
            seg_mask = seg_mask.type(torch.long)

        sample = {"img": img, "mask": seg_mask, "filename": self.dataset[idx]}

        return sample


if __name__=="__main__":
    transforms = A.Compose([
        A.Resize(874//2, 1164//2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    dataset = Comma10kDataset("/home/martin/datasets/comma10k", "train", transforms)
    print(len(dataset))
    sample = dataset[1]
    print(sample.keys())
    print(sample['img'].shape)
    print(sample['mask'].shape)

    from matplotlib import pyplot as plt
    img = utils.img_display(sample['img'])
    mask = utils.mask_display(sample['mask'])
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(mask)
    plt.show()
