import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from advanced_model import AdvancedModel
from gradcam import GradCAM
from inference import predict_image, mc_dropout_predict, interpret_uncertainty

# 🔥 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📥 Load model
model = AdvancedModel(num_classes=4)
model.load_state_dict(torch.load("advanced_model.pth"))
model.to(device)
model.eval()

# 🎯 GradCAM target layer
target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

# 🖼️ Load image
image_path = "test_image.jpg"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# =========================
# 🔮 PREDICTION + UNCERTAINTY
# =========================

label, confidence = predict_image(model, image_tensor)
_, _, unc = mc_dropout_predict(model, image_tensor)
unc_text = interpret_uncertainty(unc)

# =========================
# 🔥 GRAD-CAM
# =========================

cam = gradcam.generate(image_tensor)

# 🖼️ Original image
original = cv2.imread(image_path)
original = cv2.resize(original, (224,224))

# Heatmap + overlay
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# =========================
# 🧾 DICOM-STYLE TEXT (CLEAN)
# =========================

def put_text_dicom(img, text, position):
    x, y = position
    
    # Shadow
    cv2.putText(img, text, (x+1, y+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 0), 2, cv2.LINE_AA)
    
    # Main text
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)

h, w, _ = overlay.shape

# 🔝 TOP LEFT (Diagnosis only)
put_text_dicom(overlay, f"DX: {label.upper()}", (8, 15))

# 🔻 BOTTOM LEFT (Confidence)
put_text_dicom(overlay, f"Conf: {confidence*100:.2f}%", (8, h - 10))

# 🔻 BOTTOM RIGHT (Uncertainty - aligned properly)
unc_text_full = f"Unc: {unc_text}"
(text_w, _), _ = cv2.getTextSize(unc_text_full, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

put_text_dicom(overlay, unc_text_full, (w - text_w - 8, h - 10))
# =========================
# 💾 SAVE OUTPUT
# =========================

cv2.imwrite("plots/gradcam_overlay.jpg", overlay)

# =========================
# 🖥️ TERMINAL OUTPUT
# =========================

print("----- FINAL RESULT -----")
print(f"Prediction: {label}")
print(f"Confidence: {confidence*100:.2f}%")
print(f"Uncertainty: {unc_text}")
print("✅ Final output saved as final_output.jpg")