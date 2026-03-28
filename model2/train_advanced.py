# Model 2: Advanced Model using EfficientNet, GradCAM, and Attention
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from advanced_model import AdvancedModel
from data_loader import get_data_loaders

# 🔥 GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 📂 Paths
train_dir = "../dataset/Training"
val_dir = "../dataset/Testing"

train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size=16)

# 🧠 Model
model = AdvancedModel(num_classes=4).to(device)

# Freeze early layers
for param in model.features[:-2].parameters():
    param.requires_grad = False

# Unfreeze last layers
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 🔥 TRACKING LISTS
train_losses = []
val_losses = []
train_accs = []
val_accs = []

epochs = 20

for epoch in range(epochs):

    # 🔵 TRAINING
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    avg_train_loss = total_loss / len(train_loader)

    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    # 🟢 VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    # 🔥 Print logs
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Train Acc: {train_acc:.4f} "
          f"Val Acc: {val_acc:.4f}")

    scheduler.step()

# 💾 Save model
torch.save(model.state_dict(), "advanced_model.pth")

# 💾 Save training history
history = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_accs,
    "val_acc": val_accs
}

with open("training_history.pkl", "wb") as f:
    pickle.dump(history, f)

print("✅ Training complete. Model + history saved!")