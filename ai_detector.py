import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import models, transforms

# --- CONFIGURATION ---
BATCH_SIZE = 16 
IMG_SIZE = 224
EPOCHS = 3 
MODEL_NAME = "resnet_ai_detector.pth"

# --- TRANSFORMS ---
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- DATASET CLASS ---
class AIDetectorDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']

        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# --- MODEL BUILDING ---
def build_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last layer for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model

# --- HELPERS ---
def get_data_df(base_path):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    data = []
    fake_dir = base_path / "FAKE1"
    real_dir = base_path / "REAL1"

    if fake_dir.exists():
        for ext in extensions:
            for img_path in fake_dir.glob(ext):
                data.append([img_path, 1])

    if real_dir.exists():
        for ext in extensions:
            for img_path in real_dir.glob(ext):
                data.append([img_path, 0])

    return pd.DataFrame(data, columns=['path', 'label'])

# --- TRAINING RUN ---
def run_app():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_resnet().to(device)

    base_data_path = Path("DATA_COMBINED")
    train_path = base_data_path / "train"
    val_path = base_data_path / "val"

    train_df = get_data_df(train_path).sample(frac=1).reset_index(drop=True)
    val_df = get_data_df(val_path).sample(frac=1).reset_index(drop=True)

    if train_df.empty or val_df.empty:
        print("ERROR: DATA_COMBINED must contain train/val folders with FAKE1/REAL1 subfolders")
        return

    # Limit data for demo purposes
    train_df = train_df.head(5000)
    val_df = val_df.head(5000)

    train_loader = DataLoader(AIDetectorDataset(train_df, train_transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AIDetectorDataset(val_df, val_transforms), batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_NAME)
    print(f"Model saved to {MODEL_NAME}")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_app()