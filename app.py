import streamlit as st
import requests
import io
import base64
from PIL import Image, ImageOps
import os

# Page configuration
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #358ccb;
        transform: scale(1.02);
    }
    .status-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #161b22;
        border-left: 5px solid #1f77b4;
        margin-bottom: 2rem;
    }
    h1 {
        color: #e6edf3;
    }
    h2, h3 {
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# App branding
st.title("👷 Helmet Detection System")
st.subheader("Powered by YOLOv8 & FastAPI")

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", use_container_width=True)
    st.header("⚙️ Configuration")
    api_url = st.text_input("Backend API URL", value="http://localhost:8000/predict")
    st.info("Ensure the FastAPI backend is running before submitting images.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        original_image = ImageOps.exif_transpose(original_image)
        st.image(original_image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🚀 Detect Helmet"):
            with st.spinner("Processing image via YOLOv8 backend..."):
                try:
                    # Prepare the file for the request
                    # Need to seek back to start since st.image might have read it
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    
                    # Call API
                    response = requests.post(api_url, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["api_response"] = data
                    else:
                        st.error(f"Error: Backend returned status {response.status_code}")
                        st.write(response.text)
                except Exception as e:
                    st.error(f"Failed to connect to backend: {str(e)}")

with col2:
    st.header("📊 Detection Result")
    
    if "api_response" in st.session_state:
        data = st.session_state["api_response"]
        
        if data["success"]:
            # Decode the base64 image
            img_data = base64.b64decode(data["image_data"])
            processed_image = Image.open(io.BytesIO(img_data))
            
            st.image(processed_image, caption="Processed Image", use_container_width=True)
            
            # Display detections
            st.write(f"### {data['message']}")
            
            detections = data["detections"]
            if detections:
                for d in detections:
                    label = d["label"]
                    conf = d["confidence"]
                    st.markdown(f"- **{label.upper()}**: Confidence {conf:.2f}")
            else:
                st.warning("No helmets or persons detected.")
        else:
            st.error(f"API Error: {data.get('error', 'Unknown error')}")
    else:
        st.info("Awaiting detection result. Upload an image and click 'Detect Helmet'.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #8b949e;'>
    Built with ❤️ using Streamlit, FastAPI, and Ultralytics YOLOv8
</div>
""", unsafe_allow_html=True)
