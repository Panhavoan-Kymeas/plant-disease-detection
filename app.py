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


disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot is a common disease in bell peppers caused by Xanthomonas bacteria, resulting in dark, water-soaked lesions on leaves and fruits.",
        "treatment": "Apply copper-based bactericides and remove infected plant parts to reduce spread.",
        "prevention": "Plant resistant varieties, rotate crops, and avoid overhead irrigation."
    },
    "Pepper__bell___healthy": {
        "description": "The pepper plant appears healthy with no signs of disease.",
        "treatment": "No treatment required.",
        "prevention": "Maintain good cultural practices, regular monitoring, and proper fertilization."
    },
    "Potato___Early_blight": {
        "description": "Early blight is caused by the fungus Alternaria solani and leads to brown concentric lesions on older leaves.",
        "treatment": "Apply fungicides like chlorothalonil and remove affected leaves.",
        "prevention": "Rotate crops, avoid wetting leaves during irrigation, and plant resistant varieties."
    },
    "Potato___Late_blight": {
        "description": "Late blight is caused by Phytophthora infestans, affecting leaves and tubers, leading to dark lesions and rot.",
        "treatment": "Use fungicides like mancozeb at early infection stages.",
        "prevention": "Ensure good drainage, remove infected debris, and plant resistant varieties."
    },
    "Potato___healthy": {
        "description": "The potato plant is healthy with no detectable disease symptoms.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper crop rotation and monitor plants regularly."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot affects tomato leaves and fruits, causing small, dark lesions.",
        "treatment": "Use copper-based bactericides and remove infected leaves.",
        "prevention": "Plant disease-free seeds and practice crop rotation."
    },
    "Tomato_Early_blight": {
        "description": "Early blight in tomatoes is caused by Alternaria solani, producing dark lesions on leaves and stems.",
        "treatment": "Apply fungicides and prune infected leaves.",
        "prevention": "Use resistant varieties and practice crop rotation."
    },
    "Tomato_Late_blight": {
        "description": "Late blight is a severe fungal disease caused by Phytophthora infestans affecting leaves, stems, and fruits.",
        "treatment": "Spray fungicides such as chlorothalonil or mancozeb and remove infected plant parts.",
        "prevention": "Ensure good air circulation, avoid overhead watering, and remove crop debris."
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold in tomatoes is caused by the fungus Passalora fulva, resulting in yellow spots on leaves with mold on the underside.",
        "treatment": "Use fungicides and remove severely infected leaves.",
        "prevention": "Avoid dense planting, improve ventilation, and water at the base of plants."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot is caused by Septoria lycopersici, producing small circular spots with dark borders on leaves.",
        "treatment": "Apply fungicides and remove infected foliage.",
        "prevention": "Use clean seeds, rotate crops, and avoid wetting leaves during irrigation."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Two-spotted spider mites infest tomato plants, causing yellow speckling and webbing on leaves.",
        "treatment": "Use miticides or insecticidal soaps and remove heavily infested leaves.",
        "prevention": "Maintain humidity, regularly inspect plants, and introduce natural predators like ladybugs."
    },
    "Tomato__Target_Spot": {
        "description": "Target spot causes concentric dark spots on leaves, stems, and fruits, often leading to defoliation.",
        "treatment": "Apply appropriate fungicides and remove infected plant debris.",
        "prevention": "Rotate crops and plant resistant varieties."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Tomato Yellow Leaf Curl Virus causes yellowing, curling, and stunted growth in tomato plants.",
        "treatment": "No chemical treatment; remove infected plants immediately.",
        "prevention": "Control whiteflies, plant resistant varieties, and remove weeds that host the virus."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Tomato Mosaic Virus causes mottled leaves, leaf distortion, and reduced fruit yield.",
        "treatment": "Remove and destroy infected plants; no chemical cure.",
        "prevention": "Use virus-free seeds, disinfect tools, and practice crop rotation."
    },
    "Tomato_healthy": {
        "description": "The tomato plant is healthy with no detectable disease symptoms.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain good agricultural practices and monitor plants regularly."
    }
}


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
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {"class": class_names[idx], "confidence": float(predictions[0][idx]*100)}
        for idx in top_3_idx
    ]

    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[0][predicted_idx]*100)

    # Add disease info if available
    info = disease_info.get(predicted_class, {
        "description": "Information not available.",
        "treatment": "N/A",
        "prevention": "N/A"
    })

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_3_predictions": top_3_predictions,
        "description": info["description"],
        "treatment": info["treatment"],
        "prevention": info["prevention"]
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
