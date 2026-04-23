# ============================================================
# FastAPI Inference Server for Human Activity Recognition
#
# Loads the quantized ONNX model and serves predictions
# via a REST API endpoint.
#
# Usage:
#   Start server:  uvicorn src.app:app --reload
#   Send request:  POST http://localhost:8000/predict
#                  with JSON body containing sensor data
# ============================================================

from fastapi import FastAPI
import numpy as np
import onnxruntime as ort
import os

# Initialize FastAPI app
app = FastAPI(
    title="HAR Prediction API",
    description="Classifies human activities from smartphone sensor data"
)

# Load quantized ONNX model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'har_cnn_quantized.onnx')
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Load training set statistics for normalization
# These must match what was used during training
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'UCI HAR Dataset')
def load_signals(subset):
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    signals = []
    for signal in signal_types:
        filepath = os.path.join(DATA_PATH, subset, 'Inertial Signals', f'{signal}_{subset}.txt')
        data = np.genfromtxt(filepath)
        signals.append(data)
    return np.stack(signals, axis=-1)

train_data = load_signals('train')
TRAIN_MEAN = train_data.mean(axis=(0, 1))
TRAIN_STD = train_data.std(axis=(0, 1))

# Activity labels
ACTIVITIES = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

@app.get("/")
def home():
    """Health check endpoint."""
    return {"status": "running", "model": "HAR CNN (quantized)"}

@app.post("/predict")
def predict(data: dict):
    """
    Predict activity from sensor data.
    
    Expected input format:
    {
        "sensor_data": [[9 values] x 128 timesteps]
    }
    Shape: (128, 9) - 128 timesteps, 9 sensor channels
    """
    try:
        # Convert input to numpy array
        sensor_data = np.array(data["sensor_data"], dtype=np.float32)
        
        # Reshape to (1, 9, 128) - batch, channels, timesteps
        if sensor_data.shape == (128, 9):
            sensor_data = sensor_data.T[np.newaxis, :, :]
        elif sensor_data.shape == (9, 128):
            sensor_data = sensor_data[np.newaxis, :, :]
        else:
            return {"error": f"Expected shape (128, 9) or (9, 128), got {sensor_data.shape}"}
        
        # Normalize using training statistics
        for ch in range(9):
            sensor_data[0, ch, :] = (sensor_data[0, ch, :] - TRAIN_MEAN[ch]) / TRAIN_STD[ch]
        

        # Run inference
        outputs = session.run(None, {input_name: sensor_data})
        probs = outputs[0][0]
        
        # Get prediction
        pred_idx = int(np.argmax(probs))
        confidence = float(np.exp(probs[pred_idx]) / np.sum(np.exp(probs)))
        
        return {
            "activity": ACTIVITIES[pred_idx],
            "confidence": round(confidence, 4),
            "all_probabilities": {
                ACTIVITIES[i]: round(float(np.exp(probs[i]) / np.sum(np.exp(probs))), 4)
                for i in range(6)
            }
        }
    except Exception as e:
        return {"error": str(e)}