# utils/preprocessing.py
import cv2
import numpy as np

def preprocess_image(image):
    resized = cv2.resize(image, (128, 128))  # Match original TF size
    normalized = resized / 255.0
    flat = normalized.flatten().reshape(1, -1)
    return flat
