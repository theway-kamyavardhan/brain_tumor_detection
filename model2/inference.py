import torch
import numpy as np
import torch.nn.functional as F

from advanced_model import AdvancedModel

# 🔥 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Class names
class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']

# 📥 Load model
model = AdvancedModel(num_classes=4)
model.load_state_dict(torch.load("advanced_model.pth"))
model.to(device)
model.eval()


# =========================================
# 🔮 NORMAL PREDICTION (CLASS + CONFIDENCE)
# =========================================
def predict_image(model, image_tensor):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

        confidence, pred_class = torch.max(probs, 1)

    label = class_names[pred_class.item()]
    confidence = confidence.item()

    return label, confidence


# =========================================
# 🔥 MC DROPOUT (UNCERTAINTY)
# =========================================
def mc_dropout_predict(model, image_tensor, n_samples=20):
    image_tensor = image_tensor.to(device)

    model.train()  # keep dropout ON

    preds = []

    for _ in range(n_samples):
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        preds.append(probs.detach().cpu().numpy())

    preds = np.array(preds)

    mean_pred = preds.mean(axis=0)
    uncertainty = preds.var(axis=0)

    # Get final class
    pred_class = np.argmax(mean_pred)
    label = class_names[pred_class]
    confidence = mean_pred[0][pred_class]
    uncertainty_score = uncertainty[0][pred_class]

    return label, confidence, uncertainty_score


# =========================================
# 🎯 HUMAN-READABLE OUTPUT
# =========================================
def interpret_uncertainty(score):
    if score < 0.01:
        return "Low"
    elif score < 0.05:
        return "Medium"
    else:
        return "High"