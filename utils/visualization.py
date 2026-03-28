import os
import sys
import cv2
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model2')))

from model2.gradcam import GradCAM

# plot the class probabilities as a horizontal bar chart
def plot_probability_distribution(probabilities, title="Probability Distribution", color="#007AFF"):
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes] # Convert to percentage

    # Sort
    sorted_indices = np.argsort(probs)
    sorted_classes = [classes[i].capitalize() for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]

    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_classes,
        orientation='h',
        marker_color=color,
        text=[f"{p:.1f}%" for p in sorted_probs],
        textposition='outside',
        cliponaxis=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Confidence (%)",
        yaxis_title="Class",
        xaxis=dict(range=[0, 110], showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        height=250,
        font=dict(size=14)
    )
    
    return fig

# generate grad-cam heatmap overlay
def generate_gradcam_overlay(image_tensor, model, original_image):
    # Specifically for efficientnet base in AdvancedModel
    target_layer = model.features[-1]
    
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image_tensor)
    
    # Original image resizing
    if isinstance(original_image, str):
        original = cv2.imread(original_image)
    elif isinstance(original_image, Image.Image):
        # Convert PIL to cv2 (BGR)
        original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    else:
        original = original_image
        
    original = cv2.resize(original, (224, 224))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    
    # Convert BGR back to RGB for display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb
