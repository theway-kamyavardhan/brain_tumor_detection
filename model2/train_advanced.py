# training logic for model 2
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from advanced_model import AdvancedModel
from data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# data setup
train_dir = "../dataset/Training"
val_dir = "../dataset/Testing"

train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size=16)

model = AdvancedModel(num_classes=4).to(device)

# fine-tuning setup
for param in model.features[:-2].parameters():
    param.requires_grad = False

for param in model.features[-2:].parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
# Adam with Weight Decay (L2 Regularization)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4, 
    weight_decay=1e-4
)

# 🚦 ADAPTIVE LR Scheduler (Respond to validation loss)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3, 
    verbose=True
)

# stats tracking
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_loss = float('inf')

epochs = 20

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=5)

for epoch in range(epochs):

    # 🔵 TRAINING
    model.train()
    total_loss, correct, total = 0, 0, 0

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
    val_correct, val_total, val_loss = 0, 0, 0

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

    # 🚥 Step the Adaptive Scheduler
    scheduler.step(avg_val_loss)

    # 🏆 Best Model Checkpoint (Save only if it actually got better)
    if avg_val_loss < best_val_loss:
        print(f"Validation Loss improved: {best_val_loss:.4f} -> {avg_val_loss:.4f}. Saving best model.")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "advanced_model.pth")

    # save history
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    }

    with open("training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    # Check Early Stopping
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered. Training finished.")
        break

    print("Epoch finished.")