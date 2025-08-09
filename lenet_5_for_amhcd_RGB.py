import os
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from tqdm import tqdm

CONFIG = {
    'data_dir': r'D:/Cours/S2/Deep Learning/Devoirs/TP8/amhcd-data-64/tifinagh-images',  # Change to your dataset path
    'output_dir': 'outputs',
    'batch_size': 64,
    'epochs': 30,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'val_split': 0.2,
    'num_workers': 4,
    'input_size': 32,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

class LeNet5(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

def get_dataloaders(data_dir, input_size=32, batch_size=64, val_split=0.2, num_workers=4, seed=42):
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")
    train_tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_dir = Path(data_dir)
    if (data_dir / 'train').exists():
        train_dataset = datasets.ImageFolder(data_dir / 'train', transform=train_tfm)
        if (data_dir / 'val').exists():
            val_dataset = datasets.ImageFolder(data_dir / 'val', transform=val_tfm)
        else:
            val_size = int(len(train_dataset) * val_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
            val_dataset.dataset.transform = val_tfm
    else:
        full_dataset = datasets.ImageFolder(data_dir, transform=train_tfm)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
        val_dataset.dataset.transform = val_tfm
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = sorted(train_dataset.dataset.class_to_idx, key=train_dataset.dataset.class_to_idx.get) if isinstance(train_dataset, torch.utils.data.Subset) else sorted(train_dataset.class_to_idx, key=train_dataset.class_to_idx.get)
    return train_loader, val_loader, classes

def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in tqdm(dataloader, desc='Train', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)
    return running_loss / total, correct / total

def eval_model(model, device, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Val', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
    return running_loss / total, correct / total

def fit(model, train_loader, val_loader, device, epochs=30, lr=1e-3, weight_decay=1e-4, output_dir='outputs'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_model(model, device, val_loader, criterion)
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f" train_loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f" val_loss:   {val_loss:.4f} acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({'model_state_dict': best_model_wts, 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'best_acc': best_acc}, os.path.join(output_dir, 'best_model.pth'))
            print(' Saved best model')
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(output_dir, 'lenet_final.pth'))
    return model, history

def plot_history(history, output_dir='outputs'):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.show()

if __name__ == '__main__':
    print('Device:', CONFIG['device'])
    train_loader, val_loader, classes = get_dataloaders(CONFIG['data_dir'], CONFIG['input_size'], CONFIG['batch_size'], CONFIG['val_split'], CONFIG['num_workers'], CONFIG['seed'])
    print('Classes ({}):'.format(len(classes)), classes)
    model = LeNet5(in_channels=3, num_classes=len(classes))
    model, history = fit(model, train_loader, val_loader, CONFIG['device'], CONFIG['epochs'], CONFIG['lr'], CONFIG['weight_decay'], CONFIG['output_dir'])
    plot_history(history, CONFIG['output_dir'])
    print('Done. Models and logs saved to', CONFIG['output_dir'])
