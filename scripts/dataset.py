# scripts/dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Food101Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Read class names
        self.classes = sorted(os.listdir(os.path.join(root_dir, "images")))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load train/test split from txt
        split_file = os.path.join(root_dir, "meta", f"{split}.txt")
        with open(split_file, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            cls = line.split("/")[0]
            path = os.path.join(root_dir, "images", line + ".jpg")
            self.image_paths.append(path)
            self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
