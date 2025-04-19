# scripts/train_classifier.py

import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import Food101Dataset
import os
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load small subset for fast debug
    full_train = Food101Dataset(root_dir="data/food-101", split="train", transform=train_transform)
    full_val = Food101Dataset(root_dir="data/food-101", split="test", transform=val_transform)

    train_dataset = Subset(full_train, list(range(0, 200)))
    val_dataset = Subset(full_val, list(range(0, 100)))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Model
    model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=101)
    model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 🔥 Fast training loop
    print("🔥 Starting quick debug training (1 epoch, 200 samples)...")
    for epoch in range(1):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 5 == 0:
                print(f"Epoch {epoch+1} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} finished. Avg Loss: {running_loss / len(train_loader):.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/debug_efficientnetv2_food101.pth")
    print("✅ Model saved: models/debug_efficientnetv2_food101.pth")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
