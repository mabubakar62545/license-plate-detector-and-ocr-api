# ============================================================================
# IMPORTS
# ============================================================================
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List

import cv2
import numpy as np
import os
import tempfile
import shutil

from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from fast_plate_ocr import LicensePlateRecognizer

TEMP_DIR = tempfile.mkdtemp()


# ============================================================================
# RESPONSE MODELS
# ============================================================================
class Detection(BaseModel):
    plate_text: str
    bbox: List[float]
    confidence: float
    ocr_confidence: float

class Response(BaseModel):
    success: bool
    total_detections: int
    detections: List[Detection]


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print("Loading models...")
ocr_model = LicensePlateRecognizer('cct-xs-v1-global-model')
detection_model = YOLO('models/CarNumberPlateDetector.pt')
print("Models loaded!")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_image(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return img


def process_video_file_sync(input_path: str, output_path: str, conf: float):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Could not open video file.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    track_history = {}

    while True:
        ret, im0 = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run Detection & OCR
        results = detection_model.track(im0, persist=True, conf=conf) 
        
        if results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                crop = im0[y1:y2, x1:x2]
                
                plate_text = "N/A"
                if crop.size > 0:
                    try:
                        plates, _ = ocr_model.run(crop, return_confidence=True)
                        if plates:
                            plate_text = plates[0]
                    except:
                        pass
                
                # Draw Bounding Box and Label
                track_color = colors(int(track_id), True) 
                label = f"ID:{track_id} | {plate_text}"
                cv2.rectangle(im0, (x1, y1), (x2, y2), track_color, 2)
                
                # Draw text background
                (w_label, h_label), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(im0, (x1, y1 - h_label - 10), (x1 + w_label, y1), track_color, -1)
                cv2.putText(im0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        out.write(im0)

    out.release()
    cap.release()
    return output_path, frame_count


def file_iterator(file_path: str, chunk_size: int = 1024 * 1024):
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    finally:
        os.unlink(file_path)


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(title="License Plate Detection API", version="1.0.0")

# Add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {"message": "License Plate Detection API", "status": "running"}


@app.post("/read-number-plate", response_model=Response)
async def detect(file: UploadFile = File(...), conf: float = 0.3):
    """
    Endpoint 1: Detect and read license plates from image
    Returns: JSON with detection data only
    """
    
    # Read image
    img = read_image(file)
    
    # Run detection
    results = detection_model.predict(img, conf=conf)
    
    detections = []
    
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        for box, confidence in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            
            # Crop plate region
            crop = img[y1:y2, x1:x2]
            
            # Run OCR
            plate_text = "N/A"
            ocr_conf = 0.0
            
            if crop.size > 0:
                try:
                    plates, confidences = ocr_model.run(crop, return_confidence=True)
                    if plates:
                        plate_text = plates[0]
                        ocr_conf = float(confidences[0]) if confidences else 0.0
                except:
                    pass
            
            detections.append(Detection(
                plate_text=plate_text,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(confidence),
                ocr_confidence=ocr_conf
            ))
    
    return Response(
        success=True,
        total_detections=len(detections),
        detections=detections
    )


@app.post("/process-image")
async def process_image_endpoint(file: UploadFile = File(...), conf: float = 0.3):
    """
    Endpoint 2: Process image and return annotated image for download
    Returns: Annotated image file with bounding boxes and plate text
    """
    # Read image
    img = read_image(file)
    img_annotated = img.copy()
    
    # Run detection
    results = detection_model.predict(img, conf=conf)
    
    detection_count = 0
    
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        for idx, (box, confidence) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            
            # Crop plate region
            crop = img[y1:y2, x1:x2]
            
            # Run OCR
            plate_text = "N/A"
            ocr_conf = 0.0
            
            if crop.size > 0:
                try:
                    plates, confidences = ocr_model.run(crop, return_confidence=True)
                    if plates:
                        plate_text = plates[0]
                        ocr_conf = float(confidences[0]) if confidences else 0.0
                except:
                    pass
            
            # Draw Bounding Box and Label
            box_color = colors(idx, True)
            label = f"{plate_text} | Conf: {confidence:.2f}"
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw text background
            (w_label, h_label), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_annotated, (x1, y1 - h_label - 10), (x1 + w_label, y1), box_color, -1)
            cv2.putText(img_annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            detection_count += 1
    
    # Save annotated image to temporary file
    output_filename = os.path.join(TEMP_DIR, f"processed_{file.filename}")
    cv2.imwrite(output_filename, img_annotated)
    
    # Stream the processed image back
    response = StreamingResponse(
        file_iterator(output_filename), 
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f"attachment; filename=\"processed_{file.filename}\"",
            "X-Total-Detections": str(detection_count)
        }
    )
    
    return response


@app.post("/process-video")
async def process_video_endpoint(file: UploadFile = File(...), conf: float = 0.3):
    """
    Endpoint 3: Process video and return annotated video for download
    Returns: Annotated video file with tracking IDs and plate text
    """
    # Save uploaded file to a temporary, named file on disk
    input_filename = os.path.join(TEMP_DIR, f"input_{file.filename}")
    output_filename = os.path.join(TEMP_DIR, f"processed_{file.filename}")

    try:
        # Asynchronously read the uploaded file contents
        contents = await file.read()
        # Synchronously write contents to disk
        with open(input_filename, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save uploaded video file.")
    finally:
        await file.close()

    # Run the CPU-bound video processing in a thread pool
    try:
        # run_in_threadpool is essential here to prevent blocking the event loop
        final_output_path, frame_count = await run_in_threadpool(
            process_video_file_sync, 
            input_filename, 
            output_filename, 
            conf
        )
        
    except IOError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error during video processing.")
    finally:
        if os.path.exists(input_filename):
            os.unlink(input_filename)

    response = StreamingResponse(
        file_iterator(final_output_path), 
        media_type="video/mp4",
        headers={
            "Content-Disposition": f"attachment; filename=\"processed_{file.filename}\"",
            "X-Total-Frames": str(frame_count)
        }
    )
    
    return response


# ============================================================================
# EVENT HANDLERS
# ============================================================================

@app.on_event("shutdown")
def cleanup_temp_dir():
    """Clean up the temp directory on shutdown"""
    try:
        shutil.rmtree(TEMP_DIR)
    except OSError as e:
        pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)