# ============================================================
# Test script for the HAR FastAPI endpoint
#
# Loads a real sample from the test set and sends it
# to the running server to verify predictions work.
# ============================================================

import requests
import numpy as np
import pandas as pd
import os

# Load a real test sample
def load_signals(subset, base_path='data/UCI HAR Dataset'):
    signal_types = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    signals = []
    for signal in signal_types:
        filepath = os.path.join(base_path, subset, 'Inertial Signals', f'{signal}_{subset}.txt')
        data = pd.read_csv(filepath, sep=r'\s+', header=None)
        signals.append(data.values)
    return np.stack(signals, axis=-1)

X_test = load_signals('test')
y_test = pd.read_csv('data/UCI HAR Dataset/test/y_test.txt', header=None).values.flatten()

activities = {1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs",
              4: "Sitting", 5: "Standing", 6: "Laying"}

# Test 3 different activities
print("Testing HAR API with real sensor data:\n")
for i in [0, 500, 2000]:
    sample = X_test[i]  # shape: (128, 9)
    true_label = activities[y_test[i]]
    
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"sensor_data": sample.tolist()}
    )
    
    result = response.json()
    print(f"Sample {i}:")
    print(f"  True activity:      {true_label}")
    print(f"  Predicted activity: {result['activity']}")
    print(f"  Confidence:         {result['confidence']:.1%}")
    print()