import torch
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from advanced_model import AdvancedModel
from data_loader import get_data_loaders

# 🔥 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📂 Load data
train_dir = "../dataset/Training"
val_dir = "../dataset/Testing"

_, val_loader = get_data_loaders(train_dir, val_dir, batch_size=16)

# 🧠 Class names
class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

# 📥 Load model
model = AdvancedModel(num_classes=4)
model.load_state_dict(torch.load("advanced_model.pth"))
model.to(device)
model.eval()

# =========================
# 📊 COLLECT PREDICTIONS
# =========================

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =========================
# 📈 CLASSIFICATION REPORT
# =========================

print("\n===== CLASSIFICATION REPORT =====\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# =========================
# 🔥 CONFUSION MATRIX
# =========================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("plots/confusion_matrix.png")
plt.show()

# =========================
# 📊 LOAD TRAINING HISTORY
# =========================

try:
    with open("training_history.pkl", "rb") as f:
        history = pickle.load(f)

    # =========================
    # 📈 ACCURACY CURVE
    # =========================

    plt.figure()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("plots/accuracy_plot.png")
    plt.show()

    # =========================
    # 📉 LOSS CURVE
    # =========================

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("plots/loss_plot.png")
    plt.show()

    print("✅ Plots saved: accuracy_plot.png, loss_plot.png")

except FileNotFoundError:
    print("⚠️ training_history.pkl not found. Run training first to generate plots.")