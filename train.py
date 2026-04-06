from ultralytics import YOLO

def train_model():
    """
    Trains the YOLOv8 model on the custom helmet detection dataset.
    Initializes with 'yolov8n.pt' and saves the best performance weights.
    """
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model
    # data: path to the data.yaml file
    # epochs: number of training iterations
    # imgsz: image size
    results = model.train(
        data="data.yaml",
        epochs=5,
        imgsz=640,
        name="helmet_detection"
    )
    
    print("Training complete. Best weights saved to runs/detect/helmet_detection/weights/best.pt")

if __name__ == "__main__":
    train_model()
