import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("mnist_cnn.h5")  # ya my_model.keras agar aapne naya format use kiya hai

def predict_digit(img_path):
    # Image read karo (grayscale me)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28 (MNIST format)
    img = cv2.resize(img, (28, 28))

    # Invert colors (MNIST me white background aur black digit hota hai)
    img = 255 - img

    # Normalize (0-1 range)
    img = img / 255.0

    # Reshape for model input (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)

    # Prediction
    pred = model.predict(img)
    digit = np.argmax(pred)

    print(f"Predicted Digit: {digit}")

# Example: yaha apni digit image ka path do
predict_digit("digit.png")
