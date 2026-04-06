from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageOps
import io
import os
import base64

app = FastAPI(title="Helmet Detection API")

# Setup directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
WEIGHTS_PATH = os.path.join(MODELS_DIR, "best.pt")
FALLBACK_WEIGHTS = "yolov8n.pt"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load the model
# Check if custom weights exist in models/, otherwise check the training run, otherwise fallback
if os.path.exists(WEIGHTS_PATH):
    model = YOLO(WEIGHTS_PATH)
    print(f"✅ Loaded premium custom weights from {WEIGHTS_PATH}")
elif os.path.exists("runs/detect/helmet_detection/weights/best.pt"):
    model = YOLO("runs/detect/helmet_detection/weights/best.pt")
    print("✅ Loaded custom weights from local training run.")
else:
    model = YOLO(FALLBACK_WEIGHTS)
    print(f"⚠️ Using fallback weights: {FALLBACK_WEIGHTS}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for helmet detection.
    Processes an uploaded image and returns detections along with the plotted image.
    """
    # 1. Validate file extension
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # 2. Read image content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image).convert("RGB") # Auto-rotate and convert
        img_np = np.array(image)
        
        # Save original image to uploads
        original_img_path = os.path.join(UPLOAD_DIR, file.filename)
        image.save(original_img_path)

        # 3. Run YOLOv8 Model
        results = model(img_np)
        
        # 4. Extract detections
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": int(box.cls[0].item()),
                "label": results[0].names[int(box.cls[0].item())],
                "confidence": float(box.conf[0].item()),
                "bbox": box.xyxy[0].tolist()
            })

        # 5. Plot results on image
        res_plotted = results[0].plot() # This returns a BGR numpy array
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Save output image
        output_img_path = os.path.join(OUTPUT_DIR, f"result_{file.filename}")
        res_image_pil = Image.fromarray(res_plotted_rgb)
        res_image_pil.save(output_img_path)

        # 6. Encode plotted image to base64 for response
        buffered = io.BytesIO()
        res_image_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "detections": detections,
            "image_data": img_base64, # Base64 encoded image
            "message": f"Detected {len(detections)} objects."
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/")
def root():
    return {"message": "Helmet Detection API is running. Use /predict to submit images."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
