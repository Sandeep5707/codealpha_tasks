import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model


# Preprocess function
def preprocess_image(path):
    img = Image.open(path).convert('L')  
    img = ImageOps.invert(img)  # white background, black digit ke liye invert required
    img = img.resize((28,28))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1,28,28,1)
    return arr

# Load trained model
model = load_model("mnist_cnn.h5")


# Folder jisme digits hain
folder = "handwritten_digits"

# Sabhi images loop me
for file in sorted(os.listdir(folder)):
    if file.endswith(".jpg"):
        path = os.path.join(folder, file)
        x = preprocess_image(path)
        pred = model.predict(x, verbose=0)
        label = np.argmax(pred)
        print(f"Image: {file}  --->  Predicted: {label}")
