from fastapi import FastAPI, UploadFile, File
import shutil
import os
from backend.api.inference import ModelInference

app = FastAPI()
model = ModelInference()  # Loads your latest_model.pth
os.makedirs("data", exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    prediction = model.predict(file_path)
    return {"prediction": prediction, "file": file.filename}

@app.get("/")
async def health_check():
    return {"message": "FastAPI server is running!"}