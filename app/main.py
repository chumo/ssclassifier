from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import cv2
import numpy as np
import shutil
import uuid

from app.schemas import DetectRequest, DetectResponse, TrainRequest
from app.geometry import get_vertices, sample_polyline_points, bilinear_interpolate
from app.utils import normalize_features
from app.classifier import predict, train_model

app = FastAPI(title="Seven-Segment OCR API")

os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Returns the visual UI for drawing coordinate boxes over the camera image."""
    ui_path = "app/static/index.html"
    if not os.path.exists(ui_path):
        return HTMLResponse(content="<h1>UI building...</h1>")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/image")
async def get_image(path: str):
    """Serve the local image path so the browser canvas can display it."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"Image not found at {req.image_path}")
        
    img = cv2.imread(req.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Failed to read image at {req.image_path}. It may not be a valid image format.")
        
    digits_count = len(req.coords) // 6
    result_str = ""
    
    for i in range(digits_count):
        idx = i * 6
        P = (req.coords[idx], req.coords[idx+1])
        X = (req.coords[idx+2], req.coords[idx+3])
        Y = (req.coords[idx+4], req.coords[idx+5])
        
        vertices = get_vertices(P, X, Y)
        sampled_points = sample_polyline_points(vertices, num_samples=80)
        
        intensities = [bilinear_interpolate(img, pt[0], pt[1]) for pt in sampled_points]
        normalized = normalize_features(np.array(intensities))
        
        try:
            digit_char = predict(normalized.tolist())
            result_str += digit_char
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed for digit {i}: {str(e)}")
            
    return DetectResponse(result=result_str)

@app.get("/trainer", response_class=HTMLResponse)
async def get_trainer_ui():
    """Returns the visual UI for creating the training dataset."""
    ui_path = "app/static/train.html"
    if not os.path.exists(ui_path):
        return HTMLResponse(content="<h1>Trainer UI building...</h1>")
    with open(ui_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Uploads an image temporarily for training."""
    upload_dir = "app/static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"path": file_path}

@app.post("/train")
async def train(req: TrainRequest):
    if not req.samples:
        raise HTTPException(status_code=400, detail="No training samples provided")
    
    X = []
    y = []
    
    for idx, sample in enumerate(req.samples):
        if not os.path.exists(sample.image_path):
            raise HTTPException(status_code=400, detail=f"Image not found at {sample.image_path}")
            
        img = cv2.imread(sample.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Failed to read image at {sample.image_path}")
            
        # Extract the 6 floats into P, X, Y vectors
        P = (sample.coords[0], sample.coords[1])
        X_vec = (sample.coords[2], sample.coords[3])
        Y_vec = (sample.coords[4], sample.coords[5])
        
        vertices = get_vertices(P, X_vec, Y_vec)
        sampled_points = sample_polyline_points(vertices, num_samples=80)
        
        intensities = [bilinear_interpolate(img, pt[0], pt[1]) for pt in sampled_points]
        normalized = normalize_features(np.array(intensities))
        
        X.append(normalized.tolist())
        y.append(sample.label)
        
    try:
        samples_trained = train_model(X, y)
        return {"message": f"Successfully trained model with {samples_trained} samples."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train model: {str(e)}")
