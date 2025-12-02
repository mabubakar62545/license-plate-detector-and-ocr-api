from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import easyocr

app = FastAPI(title="License Plate Detection API")

# Load model
MODEL_PATH = "models/CarNumberPlateDetector.pt"
model = YOLO(MODEL_PATH)

# OCR reader (optional)
reader = easyocr.Reader(['en'])

@app.get("/")
def home():
    return {"message": "License Plate Detection API"}

@app.post("/detect")
async def detect_plate(
    file: UploadFile = File(...),
    conf: float = 0.5,
    return_image: bool = False
):
    """Detect license plates in uploaded image"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        results = model.predict(img, conf=conf, device='cpu')
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence
                })
        
        # Return annotated image or JSON
        if return_image:
            annotated = results[0].plot()
            _, buffer = cv2.imencode('.jpg', annotated)
            return StreamingResponse(
                io.BytesIO(buffer.tobytes()),
                media_type="image/jpeg"
            )
        
        return JSONResponse({
            "success": True,
            "count": len(detections),
            "detections": detections
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-and-ocr")
async def detect_and_ocr(
    file: UploadFile = File(...),
    conf: float = 0.5
):
    """Detect plates and extract text with OCR"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = model.predict(img, conf=conf, device='cpu')
        
        plates = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                
                # Crop plate region
                plate_img = img[y1:y2, x1:x2]
                
                # OCR
                ocr_results = reader.readtext(plate_img)
                text = " ".join([res[1] for res in ocr_results])
                
                plates.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "text": text
                })
        
        return JSONResponse({
            "success": True,
            "count": len(plates),
            "plates": plates
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/track-video")
async def track_video(file: UploadFile = File(...)):
    """Track license plates in video"""
    # Save uploaded video
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, persist=True, conf=0.5)
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            
            frame_results.append({
                "frame": frame_idx,
                "tracks": [{"id": tid, "bbox": box} 
                          for tid, box in zip(track_ids, boxes)]
            })
        
        frame_idx += 1
    
    cap.release()
    
    return JSONResponse({
        "success": True,
        "total_frames": frame_idx,
        "results": frame_results
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)