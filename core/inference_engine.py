import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# include model folders for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'model1'))
sys.path.append(os.path.join(root_dir, 'model2'))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

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

# some weirdness with model2 imports requiring local path
original_cwd = os.getcwd()
try:
    os.chdir(os.path.join(root_dir, 'model2'))
    from model2.advanced_model import AdvancedModel
finally:
    os.chdir(original_cwd)

# Class Names
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def get_model1_class(idx):
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return classes[idx]

def get_model2_class(idx):
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    return classes[idx]

class InferenceEngine:
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_models(self):
        if self.model1 is None:
            self.model1 = CNN()
            # load weights
            self.model1.load_state_dict(torch.load("model1/outputs/best_model.pth", map_location=DEVICE))
            self.model1.to(DEVICE)
            self.model1.eval()
            
        if self.model2 is None:
            self.model2 = AdvancedModel(num_classes=4)
            self.model2.load_state_dict(torch.load("model2/advanced_model.pth", map_location=DEVICE))
            self.model2.to(DEVICE)
            self.model2.eval()
            
    def _preprocess_image(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
            
        image_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        return image, image_tensor

    def predict_model1(self, image_input):
        self.load_models()
        image, image_tensor = self._preprocess_image(image_input)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model1(image_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            confidence, pred_class = torch.max(probs, 0)
            
        inference_time = time.time() - start_time
        
        prob_dist = {get_model1_class(i): probs[i].item() for i in range(len(probs))}
        
        return {
            "model": "Model 1 (CNN)",
            "prediction_idx": pred_class.item(),
            "prediction": get_model1_class(pred_class.item()),
            "confidence": confidence.item(),
            "probabilities": prob_dist,
            "inference_time": inference_time,
            "image_tensor": image_tensor
        }

    def predict_model2(self, image_input, n_samples=20):
        self.load_models()
        image, image_tensor = self._preprocess_image(image_input)
        
        start_time = time.time()
        
        # MC sampling for uncertainty
        self.model2.train()  # dropout on
        preds = []
        for _ in range(n_samples):
            output = self.model2(image_tensor)
            probs = F.softmax(output, dim=1)
            preds.append(probs.detach().cpu().numpy())
            
        self.model2.eval() # restore state
        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)
        uncertainty = preds.var(axis=0)
        
        pred_class = np.argmax(mean_pred)
        confidence = mean_pred[0][pred_class]
        uncertainty_score = uncertainty[0][pred_class]
        
        inference_time = time.time() - start_time
        
        prob_dist = {get_model2_class(i): float(mean_pred[0][i]) for i in range(len(mean_pred[0]))}
        
        if uncertainty_score < 0.01:
            unc_label = "Low"
        elif uncertainty_score < 0.05:
            unc_label = "Medium"
        else:
            unc_label = "High"
            
        return {
            "model": "Model 2 (Advanced)",
            "prediction_idx": pred_class.item(),
            "prediction": get_model2_class(pred_class.item()),
            "confidence": float(confidence),
            "probabilities": prob_dist,
            "uncertainty_score": float(uncertainty_score),
            "uncertainty_label": unc_label,
            "inference_time": inference_time,
            "image_tensor": image_tensor,
            "model_instance": self.model2,
        }
