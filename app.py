from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# -------------------------------
# Load your model and class names
# -------------------------------

MODEL_PATH = "best_plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Replace this with your actual 15 class names in the correct order
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

IMG_SIZE = (224, 224)

# -------------------------------
# Prediction function
# -------------------------------

def predict_disease_from_bytes(image_bytes):
    # Load image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)

    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {"class": class_names[idx], "confidence": float(predictions[0][idx]*100)}
        for idx in top_3_idx
    ]

    # Get predicted class
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[0][predicted_idx]*100)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_3_predictions": top_3_predictions
    }

# -------------------------------
# FastAPI app
# -------------------------------

app = FastAPI(title="Plant Disease Detection API")

@app.get("/")
def root():
    return {"message": "Plant Disease Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()

    # Run prediction
    result = predict_disease_from_bytes(image_bytes)

    return JSONResponse(content=result)
