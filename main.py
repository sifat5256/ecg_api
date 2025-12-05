from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Load your ECG model
model = load_model("ecg_model2.keras")
print("ECG model loaded successfully!")

app = FastAPI()

# Define your class names in the same order as your model output
class_names = [
    "MI Patient",        # Myocardial Infarction Patients
    "Normal",            # Normal Person
    "History MI",        # Patient with History of MI
    "Abnormal HR"        # Abnormal Heartbeat Patients
]

# Preprocess function for your ECG images
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    # Resize to model input size (update if your model input is different)
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Root endpoint
@app.get("/")
def root():
    return {"message": "ECG Image API is running!"}

# Predict endpoint
@app.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):
    # Ensure file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    
    prediction = model.predict(img_array)
    
    # Get predicted class index
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    predicted_class_name = class_names[predicted_index]
    
    # Convert probabilities to percentages
    probabilities = (prediction[0] * 100).round(2).tolist()
    
    # Return class name + probabilities
    result = {
        "predicted_class": predicted_class_name,
        "probabilities": {class_names[i]: probabilities[i] for i in range(len(class_names))}
    }
    
    return result
