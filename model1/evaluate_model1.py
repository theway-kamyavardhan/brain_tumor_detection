# Model 1 Evaluation Script
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture (must match cnn_training.py)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def evaluate():
    # Load Data
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = datasets.ImageFolder("../dataset/Testing", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes

    # Load Model
    model = CNN().to(device)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n===== MODEL 1 CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Model 1 Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/confusion_matrix.png")
    print("Confusion matrix saved to plots/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
