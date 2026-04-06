# 👷 Helmet Detection Web System (YOLOv8 + FastAPI + Streamlit)

A production-ready, end-to-end object detection system designed to identify helmets and persons in real-time or from uploaded images. This project demonstrates high-fidelity integration between a computer vision model, a robust backend, and a modern frontend UI.

## 🚀 Features

*   **Custom YOLOv8 Training**: Automated training script for fine-tuning on helmet datasets.
*   **FastAPI Backend**: High-performance API with a dedicated `/predict` endpoint for inference.
*   **Streamlit Frontend**: A premium, user-friendly interface for image uploads and result visualization.
*   **Dual Mode Weights**: Automatically uses custom `best.pt` weights if available, or falls back to the pretrained `yolov8n.pt`.
*   **Side-by-Side Visualization**: View original vs. processed images with bounding boxes and confidence scores.

## 🛠️ Tech Stack

*   **Model**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
*   **Backend**: FastAPI, Uvicorn, Python-Multipart
*   **Frontend**: Streamlit, Requests, Pillow
*   **Image Processing**: OpenCV, NumPy

## 📁 Project Structure

```text
helmet-detection/
├── api.py           # FastAPI backend server
├── app.py           # Streamlit frontend UI
├── train.py         # YOLOv8 training script
├── detect.py        # Standalone inference script
├── requirements.txt # Project dependencies
├── data.yaml        # Dataset configuration (helmet, no_helmet)
├── models/          # Persistent storage for model weights
├── uploads/         # Directory for uploaded images
├── outputs/         # Directory for processed detections
└── README.md        # Documentation
```

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Helemt-Detection
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Train the Model
If you have a custom dataset organized as per `data.yaml`:
```bash
python train.py
```
This saves the best weights to `runs/detect/helmet_detection/weights/best.pt`.

## ▶️ How to Run

### 1. Start the FastAPI Backend
```bash
uvicorn api:app --reload
```
The API will be available at `http://localhost:8000`. You can explore the `/docs` for Swagger UI.

### 2. Start the Streamlit Frontend
In a new terminal:
```bash
streamlit run app.py
```

### 3. Run Standalone Detection
To run inference on a local file without the web UI:
```bash
python detect.py --source path/to/your/image.jpg --weights yolov8n.pt
```

## 🧪 Detection Logic
The backend uses `results[0].plot()` to draw bounding boxes and labels automatically with consistent colors. Images are processed in RGB format to ensure compatibility between PIL (Frontend) and OpenCV (Inference).

## 💡 Production Considerations
*   **Folder Cleanup**: Add a scheduled task to clear `uploads/` and `outputs/`.
*   **GPU Support**: If available, YOLOv8 will automatically utilize CUDA for faster inference.
*   **Security**: In production, secure the FastAPI endpoint and validate file sizes/formats strictly.
