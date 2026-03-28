import torch
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from advanced_model import AdvancedModel
from data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load test data
train_dir = "../dataset/Training"
val_dir = "../dataset/Testing"
_, val_loader = get_data_loaders(train_dir, val_dir, batch_size=16)

class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

# load model
model = AdvancedModel(num_classes=4)
model.load_state_dict(torch.load("advanced_model.pth"))
model.to(device)
model.eval()

# run through validation set
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

print("\nModel 2 Classification Results:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# generate roc curves
y_test_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
all_probs = np.array(all_probs)
fpr, tpr, roc_auc = dict(), dict(), dict()

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'orange']
for i, color in zip(range(4), colors):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model 2 ROC Curve')
plt.legend(loc="lower right")
plt.savefig("plots/roc_curve.png")
print("ROC curve saved to plots/roc_curve.png")

# confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("plots/confusion_matrix.png")
print("Confusion matrix saved to plots/confusion_matrix.png")

# load history and plot
try:
    with open("training_history.pkl", "rb") as f:
        history = pickle.load(f)

    # accuracy curves
    plt.figure()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("plots/accuracy_plot.png")

    # loss curves
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("plots/loss_plot.png")

    print("Plots saved: accuracy_plot.png, loss_plot.png")

except FileNotFoundError:
    print("training_history.pkl not found. Run training first to generate plots.")