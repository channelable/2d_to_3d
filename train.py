import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.optim import AdamW
from data import FloorplanDataset
from model import get_model
import albumentations as A
import matplotlib.pyplot as plt

# --- CONFIG ---
IMG_DIR = "data/images"
MASK_DIR = "data/masks"
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- AUGMENTATION ---
transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)
])

# --- DATA ---
dataset = FloorplanDataset(IMG_DIR, MASK_DIR, transform)
n = len(dataset)
train_size = int(0.8 * n)
val_size = n - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --- MODEL ---
model = get_model().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=LR)

# --- TRAIN LOOP ---
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
