import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_transforms(is_training=True):
    """Get image transforms for training or evaluation."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if is_training:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


def split_cattle_ids(dataset_path=None):
    """Split cattle folder IDs into train/val/test sets by identity."""
    if dataset_path is None:
        dataset_path = config.DATASET_PATH

    # Get numbered folders only (exclude OriginalMaster)
    all_ids = sorted(
        [d for d in os.listdir(dataset_path)
         if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()],
        key=int,
    )

    random.seed(42)
    random.shuffle(all_ids)

    n = len(all_ids)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = train_end + int(n * config.VAL_SPLIT)

    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test cattle")
    return train_ids, val_ids, test_ids


class CattleMuzzleDataset(Dataset):
    """Generates positive/negative pairs of cattle muzzle images."""

    def __init__(self, dataset_path, cattle_ids, transform=None, pairs_per_epoch=None):
        self.dataset_path = dataset_path
        self.cattle_ids = cattle_ids
        self.transform = transform or get_transforms(is_training=True)
        self.pairs_per_epoch = pairs_per_epoch or config.PAIRS_PER_EPOCH

        # Build image index: {cattle_id: [image_paths]}
        self.images = {}
        total = 0
        for cid in self.cattle_ids:
            folder = os.path.join(dataset_path, cid)
            imgs = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(imgs) >= 2:
                self.images[cid] = imgs
                total += len(imgs)

        # Filter cattle_ids to only those with enough images
        self.cattle_ids = list(self.images.keys())
        print(f"Loaded {total} images from {len(self.cattle_ids)} cattle")

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        if random.random() < 0.5:
            # Positive pair: same cattle
            cid = random.choice(self.cattle_ids)
            img1_path, img2_path = random.sample(self.images[cid], 2)
            label = 1.0
        else:
            # Negative pair: different cattle
            cid1, cid2 = random.sample(self.cattle_ids, 2)
            img1_path = random.choice(self.images[cid1])
            img2_path = random.choice(self.images[cid2])
            label = 0.0

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class SingleImageDataset(Dataset):
    """Loads all images from given cattle IDs for embedding extraction."""

    def __init__(self, dataset_path, cattle_ids, transform=None):
        self.transform = transform or get_transforms(is_training=False)
        self.samples = []  # (image_path, cattle_id)

        for cid in cattle_ids:
            folder = os.path.join(dataset_path, cid)
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder, f), cid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cid = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, cid, path
