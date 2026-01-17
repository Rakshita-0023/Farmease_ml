from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ======================
# APP INIT
# ======================
app = FastAPI(title="FarmEase Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# LOAD MODEL
# ======================
MODEL_PATH = "disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ======================
# LOAD CLASS NAMES
# ======================
CLASS_NAMES = sorted(os.listdir("plantvillage dataset/color"))

# ======================
# IMAGE PREPROCESS
# ======================
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ======================
# ROUTES
# ======================
@app.get("/")
def home():
    return {"message": "FarmEase Disease Detection API running"}

@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    predictions = model.predict(image)
    class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][class_index])

    return {
        "disease": CLASS_NAMES[class_index],
        "confidence": round(confidence * 100, 2)
    }
