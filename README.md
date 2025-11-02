# Plant Disease Detection API

This is a FastAPI project for detecting plant diseases from images using
a pre-trained TensorFlow model.

## Setup

1. Clone the repository (if applicable):  
   ```bash
   git clone <your-repo-url>
   cd Backend
   ```

2. Create a virtual environment (if not already created):  
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

    - **Windows (PowerShell):**  
      `& venv\Scripts\Activate.ps1`
    - **Windows (cmd):**  
      `venv\Scripts\activate`
    - **macOS/Linux:**  
      `source venv/bin/activate`

4. Install dependencies:  
   `pip install -r requirements.txt`


## Run the API

```bash
Start the server: uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Access: API: http://127.0.0.1:8000 Swagger UI:
http://127.0.0.1:8000/docs

## API Endpoints

1. **GET /**  
   - Returns a simple message confirming the API is running.

2. **POST /predict/**  
   - Upload an image of a plant leaf to predict the disease.  
   - **Request:** `multipart/form-data` with `file` field  
   - **Response example:**  
     ```json
     {
       "predicted_class": "Tomato_Early_blight",
       "confidence": 95.3,
       "top_3_predictions": [
         { "class": "Tomato_Early_blight", "confidence": 95.3 },
         { "class": "Tomato_Late_blight", "confidence": 3.5 },
         { "class": "Tomato_healthy", "confidence": 1.2 }
       ]
     }
     ```
    

## Notes

-   Make sure best_plant_disease_model.h5 is in the same directory as
    app.py
-   For production, consider using a more robust server setup (Gunicorn,
    Docker, etc.)
