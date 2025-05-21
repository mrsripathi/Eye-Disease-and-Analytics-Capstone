
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report 
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (smaller size for faster training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset
dataset = datasets.ImageFolder(r"C:\Users\sripa\Desktop\Guvi Final Project\data (3)\dataset", transform=transform)
class_name = dataset.classes
img, lab = dataset[0]

# Dataset splitting (70% train, 15% val, 15% test)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

# DataLoader
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# CNN Model
class EyeDiseaseCNN(nn.Module):
    def __init__(self, num_class=4):
        super(EyeDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bnor1 = nn.BatchNorm2d(32)
        self.bnor2 = nn.BatchNorm2d(64)
        self.bnor3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.bnor1(self.conv1(x))))
        x = self.pool(F.relu(self.bnor2(self.conv2(x))))
        x = self.pool(F.relu(self.bnor3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Training Function
def train_model(train_loader, val_loader, model, criterion, optimizer, device, epochs=10):
    model.to(device)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_correct.float() / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_total_loss = val_loss / len(val_loader.dataset)
        val_total_acc = val_correct.float() / len(val_loader.dataset)

        print(f"Epoch: {epoch+1}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        print(f"Validation Loss: {val_total_loss:.4f}, Validation Accuracy: {val_total_acc:.4f}")

        if val_total_loss < best_loss:
            best_loss = val_total_loss
            torch.save(model.state_dict(), "eye_cnn_model.pth")

    print("Training Complete. Best Validation Loss: {:.4f}".format(best_loss))

# Evaluation Function
def model_evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_name))

# Model instantiation
model = EyeDiseaseCNN(num_class=len(class_name))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
train_model(train_loader, val_loader, model, criterion, optimizer, device, epochs=15)

# Load best model & evaluate
model.load_state_dict(torch.load("eye_cnn_model.pth"))
model_evaluate(model, test_loader, device)
