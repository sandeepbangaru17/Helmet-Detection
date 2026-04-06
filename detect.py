import argparse
from ultralytics import YOLO
import cv2
import os

def run_inference(source, weights="yolov8n.pt", output_dir="outputs"):
    """
    Runs YOLOv8 inference on a single image or video.
    
    Args:
        source (str): Path to the input image or video.
        weights (str): Path to the model weights.
        output_dir (str): Directory to save result images.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the YOLO model
    model = YOLO(weights)

    # Run inference
    results = model(source)

    # Plot results on the image
    res_plotted = results[0].plot()

    # Save output
    output_path = os.path.join(output_dir, f"result_{os.path.basename(source)}")
    cv2.imwrite(output_path, res_plotted)
    
    print(f"Detection complete. Result saved to: {output_path}")
    print(f"Detected classes: {results[0].names}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone YOLOv8 Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to input image/video")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to model weights")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()
    run_inference(args.source, args.weights, args.output)
