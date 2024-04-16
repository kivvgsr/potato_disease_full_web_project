from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the directory containing the background image
app.mount("/static", StaticFiles(directory="C:/Users/krish/OneDrive/Desktop/potato_desease"), name="static")

# Load the pre-trained model
MODEL = tf.keras.models.load_model("C:/Users/krish/OneDrive/Desktop/potato_desease/model.h5")

# Define class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Define HTML response for the root URL
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Potato Leaf Disease Classifier</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                color: #333;
                background-image: url("/static/farmer.jpg");
                background-size: cover;
                background-position: center;
                padding: 50px 0;
            }
            .container {
                max-width: 600px;
                margin: auto;
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
            }
            h1 {
                text-align: center;
                margin-bottom: 20px;
                color: #007bff;
            }
            #uploadSection {
                margin-bottom: 20px;
            }
            #uploadedImage {
                display: none;
                max-width: 100%;
                margin-top: 20px;
            }
            #predictionResult {
                display: none;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Potato Leaf Disease Classifier</h1>
            <div id="uploadSection">
                <label for="fileInput">Upload Potato Leaf Image:</label>
                <input type="file" id="fileInput" accept="image/*">
                <button id="predictButton" class="btn btn-primary mt-2">Predict</button>
            </div>
            <div id="imageDisplay">
                <img id="uploadedImage" src="#" alt="Uploaded Image">
            </div>
            <div id="predictionResult">
                <h3>Prediction:</h3>
                <p id="predictedDisease"></p>
                <p id="confidence"></p>
            </div>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const predictButton = document.getElementById('predictButton');
            const uploadedImage = document.getElementById('uploadedImage');
            const predictionResult = document.getElementById('predictionResult');
            const predictedDisease = document.getElementById('predictedDisease');
            const confidence = document.getElementById('confidence');

            fileInput.addEventListener('change', function(event) {
                const file = event.target.files[0];
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImage.src = e.target.result;
                    uploadedImage.style.display = 'block';
                };
                reader.readAsDataURL(file);
            });

            predictButton.addEventListener('click', function() {
                if (!fileInput.files[0]) {
                    alert('Please select an image to predict.');
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    predictedDisease.textContent = `Predicted Disease: ${data.class}`;
                    confidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                    predictionResult.style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
        </script>
    </body>
    </html>
    """

# Define prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    image = np.array(Image.open(BytesIO(await file.read())))
    img_batch = np.expand_dims(image, 0)
    
    # Make predictions
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    # Return prediction result
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
