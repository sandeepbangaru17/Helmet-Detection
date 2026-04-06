from ultralytics import YOLO
import torch

def train_model():

    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train the model
    # data: path to the data.yaml file
    # epochs: number of training epochs
    # imgsz: image size
    # device: 'cuda' for GPU, 'cpu' for CPU
    results = model.train(
        data="data.yaml",
        epochs=5,
        imgsz=640,
        name="helmet_detection",
        device=device
    )
    
    print("Training complete. Best weights saved to runs/detect/helmet_detection/weights/best.pt")

if __name__ == "__main__":
    train_model()
