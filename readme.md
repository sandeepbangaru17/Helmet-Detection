<div align="center">
  <h1>👷 Helmet Detection System</h1>
  <p><i>A Production-Quality Object Detection Suite Powered by YOLOv8, FastAPI, and Streamlit</i></p>

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-00629B?style=for-the-badge&logo=ultralytics&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</div>

---

## 🌟 Project Overview
This end-to-end vision application is designed to enhance industrial safety by monitoring construction sites and workspaces for safety gear compliance. Utilizing the state-of-the-art **YOLOv8** model, the system identifies helmets, safety vests, and personal protective equipment in real-time.

## 🚀 Key Features
- **⚡ Unified Orchestration**: Run the entire system (Backend + Frontend) with a single command.
- **🛠️ Custom Dataset Integration**: Fully integrated with the high-quality *Construction Site Safety* dataset from Roboflow.
- **🖼️ Universal Image Support**: Native support for JPG, PNG, WEBP, BMP, and TIFF with automatic EXIF orientation management.
- **📊 Interactive Dashboard**: A premium Streamlit UI providing side-by-side detection comparisons and confidence analytics.
- **⚙️ Standalone CLI**: Direct inference scripts for batch processing of local images and videos.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Model** | Ultralytics YOLOv8 | Core object detection and inference |
| **Backend** | FastAPI / Uvicorn | High-performance asynchronous API services |
| **Frontend** | Streamlit | Modern, interactive web-based user interface |
| **Data** | Roboflow API | Automated acquisition of safety datasets |
| **Processing** | OpenCV / Pillow | Advanced image handling and metadata extraction |

---

## 📁 System Architecture
```text
helmet-detection/
├── main.py           # Single-entry orchestration script (Starts both servers)
├── api.py            # FastAPI backend (Inference & Media logic)
├── app.py            # Streamlit frontend (Dashboard & UI)
├── train.py          # Custom model training infrastructure
├── detect.py         # Standalone command-line inference tool
├── download_data.py  # Automated Roboflow dataset acquisition
├── data.yaml         # YOLOv8 class and path configuration
├── requirements.txt  # Project dependency manifest
├── README.md         # Professional documentation
└── datasets/         # Local storage for training data (Git ignored)
```

---

## ⚙️ Getting Started

### 1. Installation
Clone the repository and install the verified dependencies in your environment:
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup (One-time)
If you wish to download the training dataset automatically:
1. Obtain your Roboflow API Key.
2. Run the provided download tool:
```bash
python download_data.py
```

### 3. Launching the Application
Launch both the **FastAPI Backend** and the **Streamlit Dashboard** simultaneously with the unified command:
```bash
python main.py
```
> [!TIP]
> The system will automatically open your web browser to `http://localhost:8501` once both servers have initialized!

---

## 🔍 Training Your Own Model
To fine-tune the YOLOv8 model specifically for your local data:
```bash
python train.py
```
This will generate optimized weights in `runs/detect/helmet_detection/weights/best.pt`, which the system will prioritize automatically.

---

<div align="center">
  <p>Built with ❤️ for Industrial Safety and Computer Vision Excellence</p>
</div>
