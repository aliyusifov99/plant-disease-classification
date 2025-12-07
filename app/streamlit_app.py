import streamlit as st
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import PlantDiseaseClassifier


# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load model once and cache it."""
    return PlantDiseaseClassifier("efficientnet_b0")


def main():
    # Header
    st.title("🌿 Plant Disease Classification")
    st.markdown("""
    This application uses a deep learning model (EfficientNet-B0) trained on the PlantVillage dataset 
    to identify plant diseases from leaf images. Upload an image to get started!
    """)
    
    st.divider()
    
    # Load model
    with st.spinner("Loading model..."):
        classifier = load_model()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        if uploaded_file is not None:
            # Run prediction
            with st.spinner("Analyzing image..."):
                result = classifier.predict(image)
            
            # Display results
            if result['is_healthy']:
                st.success(f"✅ **Healthy Plant Detected!**")
            else:
                st.error(f"⚠️ **Disease Detected!**")
            
            # Main prediction
            st.markdown("### Diagnosis")
            st.markdown(f"**Plant:** {result['plant']}")
            st.markdown(f"**Condition:** {result['disease']}")
            st.markdown(f"**Confidence:** {result['confidence']:.2f}%")
            
            # Confidence bar
            st.progress(result['confidence'] / 100)
            
            # Disease information
            st.markdown("### 📋 Information")
            info = classifier.get_disease_info(result['predicted_class'])
            
            with st.expander("View Details", expanded=True):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Symptoms:** {info['symptoms']}")
                st.markdown(f"**Treatment:** {info['treatment']}")
            
            # Top 5 predictions
            st.markdown("### 📊 Top 5 Predictions")
            for pred in result['top5_predictions']:
                confidence = pred['confidence']
                class_name = pred['class'].replace("___", " - ").replace("_", " ")
                st.markdown(f"**{class_name}**")
                st.progress(confidence / 100)
                st.caption(f"{confidence:.2f}%")
        else:
            st.info("👈 Upload an image to see predictions")
    
    # Footer
    st.divider()
    
    # Model information
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        **Model Architecture:** EfficientNet-B0 with Transfer Learning
        
        **Dataset:** PlantVillage (54,303 images, 38 classes)
        
        **Training Approach:**
        1. Pre-trained on ImageNet
        2. Classifier head trained for 5 epochs
        3. Fine-tuned entire model for 10 epochs
        
        **Supported Plants:**
        - Apple, Blueberry, Cherry, Corn, Grape
        - Orange, Peach, Pepper, Potato, Raspberry
        - Soybean, Squash, Strawberry, Tomato
        
        **Performance:** ~98% accuracy on test set
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("🌱 Plant Disease Classifier")
        st.markdown("---")
        
        st.markdown("### How to Use")
        st.markdown("""
        1. Upload a clear image of a plant leaf
        2. Wait for the model to analyze
        3. View the diagnosis and recommendations
        """)
        
        st.markdown("---")
        st.markdown("### Tips for Best Results")
        st.markdown("""
        - Use well-lit images
        - Focus on a single leaf
        - Ensure the leaf fills most of the frame
        - Avoid blurry images
        """)
        
        st.markdown("---")
        st.markdown("### Project Info")
        st.markdown("""
        - **Course:** Deep Learning
        - **Model:** CNN + Transfer Learning
        - **Framework:** PyTorch
        """)


if __name__ == "__main__":
    main()