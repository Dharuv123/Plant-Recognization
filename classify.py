import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import numpy as np
from PIL import Image
import os

def classify_image(image_path):
    # Load the trained model
    model = load_model('retrained_model.h5')

    # Load and preprocess the image
    image = Image.open(image_path).resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the class
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    print(f'Predicted Class: {predicted_class}')

if __name__ == "__main__":
    image_path = sys.argv[1]  # Pass image path as argument
    classify_image(image_path)
