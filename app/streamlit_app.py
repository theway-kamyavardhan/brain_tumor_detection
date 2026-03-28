import os
import sys
import streamlit as st
from PIL import Image
import torch
import time

# Add root folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.inference_engine import InferenceEngine
from utils.visualization import plot_probability_distribution, generate_gradcam_overlay

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Dark / Blue medical UI */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0b1121; /* deep blue/black */
        color: #e2e8f0;            /* soft white/gray */
    }
    
    /* Card headers & metrics */
    .metric-card {
        background-color: #151e32;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 24px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6; /* vivid blue */
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Badges */
    .badge-high {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    h1, h2, h3, h4, h5 {
        font-weight: 600;
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }
    
    /* Make sidebar nice */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# --- INIT STATE ---
@st.cache_resource
def get_inference_engine():
    engine = InferenceEngine()
    # Pre-load to avoid delay on first prediction
    engine.load_models()
    return engine

engine = get_inference_engine()

if "results1" not in st.session_state:
    st.session_state.results1 = None
if "results2" not in st.session_state:
    st.session_state.results2 = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_source" not in st.session_state:
    st.session_state.image_source = None

# --- SIDEBAR NAV ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3024/3024310.png", width=60)
st.sidebar.title("NeuroScan AI")
st.sidebar.markdown("<p style='color: #86868b; margin-top: -10px;'>Advanced Brain Tumor Analytics</p>", unsafe_allow_html=True)

nav = st.sidebar.radio("Navigation", ["Home", "Upload & Predict", "Model Comparison", "Grad-CAM Visualization", "Architecture & Research"])

# --- HELPERS ---
def run_analysis(image):
    with st.spinner("Processing MRI through Model 1 (CNN)..."):
        res1 = engine.predict_model1(image)
    with st.spinner("Processing MRI through Model 2 (Advanced Attention + MC Dropout)..."):
        res2 = engine.predict_model2(image)
        
    st.session_state.results1 = res1
    st.session_state.results2 = res2
    st.session_state.current_image = image
    st.success("Analysis Complete!")
    time.sleep(0.5)

# --- PAGES ---

if nav == "Home":
    st.title("Explainable Brain Tumor Classification")
    st.markdown("""
    Welcome to the NeuroScan AI Clinical Dashboard. 
    
    This platform integrates two state-of-the-art Deep Learning models for the classification of brain tumors from MRI scans:
    - **Model 1**: A baseline Convolutional Neural Network (CNN).
    - **Model 2**: An advanced EfficientNet-B0 augmented with an Attention mechanism and MC Dropout for uncertainty quantification.
    
    ### How to use
    1. Navigate to **Upload & Predict** to submit an MRI scan.
    2. Go to **Model Comparison** to review quantitative differences.
    3. Visit **Grad-CAM Visualization** to inspect explainable AI insights.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Supported Classes</div>
            <div class="metric-value">4</div>
            <p style="color: #94a3b8; font-size: 0.9rem;">Glioma, Meningioma, Pituitary, No Tumor</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Uncertainty Estimation</div>
            <div class="metric-value">Enabled</div>
            <p style="color: #94a3b8; font-size: 0.9rem;">Via Monte Carlo Dropout</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Explainability</div>
            <div class="metric-value">Grad-CAM</div>
            <p style="color: #94a3b8; font-size: 0.9rem;">Attention overlay maps</p>
        </div>
        """, unsafe_allow_html=True)

elif nav == "Upload & Predict":
    st.title("Upload & Predict")
    st.write("Upload a patient MRI scan or use the standard reference image.")
    
    input_method = st.radio("Input Method", ["Use Default Test Image", "Upload Custom MRI"])
    
    image = None
    if input_method == "Use Default Test Image":
        default_path = os.path.join(os.path.dirname(__file__), '..', 'model2', 'test_image.jpg')
        if os.path.exists(default_path):
            image = Image.open(default_path).convert("RGB")
            st.image(image, caption="Default Test Image", width=300)
        else:
            st.error(f"Default image not found at {default_path}")
    else:
        uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI", width=300)
            
    if image is not None:
        if st.button("Run Analysis", type="primary"):
            run_analysis(image)
            
    if st.session_state.results1 is not None and st.session_state.results2 is not None:
        st.markdown("---")
        st.subheader("Quick Results")
        
        r1 = st.session_state.results1
        r2 = st.session_state.results2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Model 1 (CNN)</h4>
                <div class="metric-value">{r1['prediction'].capitalize()}</div>
                <div class="badge-high">{'✔ High Confidence' if r1['confidence'] > 0.85 else '⚠ Uncertain Prediction'}</div>
                <p style="margin-top: 15px;">Confidence: <strong>{r1['confidence']*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            unc_label = r2['uncertainty_label']
            badge_class = "badge-high" if unc_label == "Low" else ("badge-medium" if unc_label == "Medium" else "badge-low")
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Model 2 (Advanced)</h4>
                <div class="metric-value">{r2['prediction'].capitalize()}</div>
                <div class="{badge_class}">Uncertainty: {unc_label}</div>
                <p style="margin-top: 15px;">Confidence: <strong>{r2['confidence']*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("Detailed Model Comparison & Graphs")
        
        more_confident = "Model 1" if r1['confidence'] > r2['confidence'] else "Model 2"
        more_stable = "Model 2"
        
        # HTML Table for Apple-style presentation in Dark Theme
        st.markdown(f"""
        <table style="width:100%; text-align:left; background-color:#151e32; color:#e2e8f0; border-radius:12px; overflow:hidden; border-collapse: collapse; border: 1px solid #1e293b; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <tr style="background-color: #0f172a; border-bottom: 2px solid #1e293b;">
                <th style="padding: 16px;">Feature</th>
                <th style="padding: 16px; color:#3b82f6;">Model 1 (CNN)</th>
                <th style="padding: 16px; color:#818cf8;">Model 2 (Advanced)</th>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 16px; font-weight:600;">Prediction</td>
                <td style="padding: 16px;">{r1['prediction'].capitalize()}</td>
                <td style="padding: 16px;">{r2['prediction'].capitalize()}</td>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 16px; font-weight:600;">Confidence</td>
                <td style="padding: 16px; {'font-weight:bold; color:#60a5fa;' if more_confident=='Model 1' else ''}">{r1['confidence']*100:.2f}%</td>
                <td style="padding: 16px; {'font-weight:bold; color:#a5b4fc;' if more_confident=='Model 2' else ''}">{r2['confidence']*100:.2f}%</td>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 16px; font-weight:600;">Inference Speed</td>
                <td style="padding: 16px;">{r1['inference_time']*1000:.1f} ms</td>
                <td style="padding: 16px;">{r2['inference_time']*1000:.1f} ms</td>
            </tr>
            <tr>
                <td style="padding: 16px; font-weight:600;">Stability / Uncertainty</td>
                <td style="padding: 16px; color:#475569;">N/A</td>
                <td style="padding: 16px; color:#34d399; font-weight:bold;">{r2['uncertainty_label']} (Score: {r2['uncertainty_score']:.4f})</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("**Probability Distributions**")
        colA, colB = st.columns(2)
        with colA:
            fig1 = plot_probability_distribution(r1['probabilities'], title="Model 1 Distribution", color="#3b82f6")
            fig1.update_layout(font_color="#f8fafc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
        with colB:
            fig2 = plot_probability_distribution(r2['probabilities'], title="Model 2 Distribution", color="#818cf8")
            fig2.update_layout(font_color="#f8fafc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
            
        st.info("💡 For attention mapping explanations, visit **Grad-CAM Visualization**.")

elif nav == "Model Comparison":
    st.title("Model Comparison")
    
    if st.session_state.results1 is None:
        st.warning("Please run an analysis in the 'Upload & Predict' tab first.")
    else:
        r1 = st.session_state.results1
        r2 = st.session_state.results2
        
        # Comparison Table
        st.markdown("### Performance Metrics")
        
        # Decide which is more confident / stable
        more_confident = "Model 1" if r1['confidence'] > r2['confidence'] else "Model 2"
        more_stable = "Model 2" # Since Model 1 doesn't have uncertainty, Model 2 is the robust one
        
        # HTML Table for Apple-style presentation in Dark Theme
        st.markdown(f"""
        <table style="width:100%; text-align:left; background-color:#151e32; color:#e2e8f0; border-radius:12px; overflow:hidden; border-collapse: collapse; border: 1px solid #1e293b; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <tr style="background-color: #0f172a; border-bottom: 2px solid #1e293b;">
                <th style="padding: 16px;">Feature</th>
                <th style="padding: 16px; color:#3b82f6;">Model 1 (CNN)</th>
                <th style="padding: 16px; color:#818cf8;">Model 2 (Advanced)</th>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 16px; font-weight:600;">Prediction</td>
                <td style="padding: 16px;">{r1['prediction'].capitalize()}</td>
                <td style="padding: 16px;">{r2['prediction'].capitalize()}</td>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 16px; font-weight:600;">Confidence</td>
                <td style="padding: 16px; {'font-weight:bold; color:#60a5fa;' if more_confident=='Model 1' else ''}">{r1['confidence']*100:.2f}%</td>
                <td style="padding: 16px; {'font-weight:bold; color:#a5b4fc;' if more_confident=='Model 2' else ''}">{r2['confidence']*100:.2f}%</td>
            </tr>
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 16px; font-weight:600;">Inference Speed</td>
                <td style="padding: 16px;">{r1['inference_time']*1000:.1f} ms</td>
                <td style="padding: 16px;">{r2['inference_time']*1000:.1f} ms</td>
            </tr>
            <tr>
                <td style="padding: 16px; font-weight:600;">Stability / Uncertainty</td>
                <td style="padding: 16px; color:#475569;">N/A</td>
                <td style="padding: 16px; color:#34d399; font-weight:bold;">{r2['uncertainty_label']} (Score: {r2['uncertainty_score']:.4f})</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.info(f"**Insights**: {more_confident} is more confident. Model 2 provides a robustness measure with {r2['uncertainty_label']} uncertainty.")
        
        st.markdown("### Probability Distributions")
        colA, colB = st.columns(2)
        with colA:
            fig1 = plot_probability_distribution(r1['probabilities'], title="Model 1 Distribution", color="#3b82f6")
            fig1.update_layout(font_color="#f8fafc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
        with colB:
            fig2 = plot_probability_distribution(r2['probabilities'], title="Model 2 Distribution", color="#818cf8")
            fig2.update_layout(font_color="#f8fafc", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

elif nav == "Grad-CAM Visualization":
    st.title("Grad-CAM Explainability")
    
    if st.session_state.results2 is None:
        st.warning("Please run an analysis in the 'Upload & Predict' tab first.")
    else:
        r2 = st.session_state.results2
        image = st.session_state.current_image
        
        st.markdown("Highlighted regions indicate where the **Model 2 (Advanced Attention)** mechanism focused its parameters to make the classification decision.")
        
        with st.spinner("Generating Grad-CAM overlay..."):
            image_tensor = r2['image_tensor']
            model_instance = r2['model_instance']
            
            overlay_rgb = generate_gradcam_overlay(image_tensor, model_instance, image)
            
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Original MRI")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("#### Grad-CAM Overlay")
            st.image(overlay_rgb, use_container_width=True, caption=f"Attention for class: {r2['prediction'].capitalize()}")
            
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 20px;">
            <h4>Diagnostic Context</h4>
            <p><strong>Diagnosis:</strong> {r2['prediction'].upper()}</p>
            <p><strong>Confidence:</strong> {r2['confidence']*100:.2f}%</p>
            <p><strong>Uncertainty Profile (MC Dropout):</strong> {r2['uncertainty_label']} variance</p>
        </div>
        """, unsafe_allow_html=True)

elif nav == "Architecture & Research":
    st.title("Architecture & Research")
    st.markdown("This section details the architectures, training metrics, and evaluations of both models for reference.")
    
    st.markdown("### Model 1: Baseline CNN")
    st.markdown("""
    Model 1 is a custom Convolutional Neural Network built from scratch. It acts as a baseline to evaluate raw performance on the dataset.
    
    **Architecture:**
    - 3 Convolutional Blocks (Conv2D -> ReLU -> MaxPool2D)
    - Fully Connected Layer (Flatten -> Dense(128) -> ReLU -> Dropout(0.5) -> Outputs)
    - Optimized with Adam (lr=0.0005) and Early Stopping.
    """)
    
    model1_plots_dir = os.path.join(os.path.dirname(__file__), '..', 'model1', 'plots')
    col1, col2 = st.columns(2)
    with col1:
        acc1 = os.path.join(model1_plots_dir, 'accuracy.png')
        if os.path.exists(acc1): st.image(acc1, caption="Model 1 Training Accuracy")
    with col2:
        loss1 = os.path.join(model1_plots_dir, 'loss.png')
        if os.path.exists(loss1): st.image(loss1, caption="Model 1 Training Loss")
        
    cm1 = os.path.join(model1_plots_dir, 'confusion_matrix.png')
    if os.path.exists(cm1): st.image(cm1, caption="Model 1 Confusion Matrix", width=500)

    st.markdown("---")
    
    st.markdown("### Model 2: Advanced Attention Model")
    st.markdown("""
    Model 2 is a state-of-the-art framework leveraging a pre-trained EfficientNet-B0 backbone.
    
    **Architecture:**
    - **Feature Extractor**: EfficientNet-B0 (pretrained on ImageNet)
    - **Self-Attention Mechanism**: A custom Attention Layer that computes context vectors across extracted sequence features, focusing the network on critical regions.
    - **MC Dropout**: Dropout is preserved during inference to run Monte Carlo sampling, providing epistemic uncertainty estimates.
    """)
    
    model2_plots_dir = os.path.join(os.path.dirname(__file__), '..', 'model2', 'plots')
    col3, col4 = st.columns(2)
    with col3:
        acc2 = os.path.join(model2_plots_dir, 'accuracy_plot.png')
        if os.path.exists(acc2): st.image(acc2, caption="Model 2 Training Accuracy")
    with col4:
        loss2 = os.path.join(model2_plots_dir, 'loss_plot.png')
        if os.path.exists(loss2): st.image(loss2, caption="Model 2 Training Loss")
        
    cm2 = os.path.join(model2_plots_dir, 'confusion_matrix.png')
    if os.path.exists(cm2): st.image(cm2, caption="Model 2 Confusion Matrix", width=500)
    
    st.markdown("<p style='text-align: center; color: #94a3b8; margin-top: 50px;'>Built with Streamlit and PyTorch</p>", unsafe_allow_html=True)
