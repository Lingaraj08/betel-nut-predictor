import numpy as np
import cv2
from PIL import Image
import joblib

class BetelNutPredictor:
    def __init__(self):
        self.model = joblib.load('model/classifier.pkl')
        self.class_names = ['pest attack', 'premature', 'rain', 'ripeness', 'sunlight dried']

    def estimate_days_since_fallen(self, predicted_class, confidence):
        """Estimate days since the betel nut fell based on its condition"""
        if predicted_class == 'premature':
            return "1-2 days (fell before maturity)"
        elif predicted_class == 'ripeness':
            return "0-1 days (fell at perfect ripeness)"
        elif predicted_class == 'rain':
            days = max(1, min(5, int(confidence / 20)))
            return f"{days-1}-{days} days (affected by rain)"
        elif predicted_class == 'sunlight dried':
            days = max(3, min(10, int(confidence / 10)))
            return f"{days-2}-{days} days (dried in sunlight)"
        elif predicted_class == 'pest attack':
            return "unknown (pest damage makes age estimation difficult)"
        return "unknown"

    def get_fall_reason(self, predicted_class, confidence):
        reasons = {
            'pest attack': "Fell due to pest infestation affecting the stem",
            'premature': "Fell before reaching maturity, possibly due to weak attachment",
            'rain': "Heavy rain and wind caused the fall",
            'ripeness': "Natural fall due to reaching full maturity",
            'sunlight dried': "Extended exposure to sun after falling"
        }
        return reasons.get(predicted_class, "Unknown reason")

    def predict(self, image):
        # Convert PIL image to CV2 format
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Preprocess image
        img = cv2.resize(image, (128, 128))
        img = img / 255.0
        flat_img = img.reshape(1, -1)

        # Get prediction probabilities
        probabilities = self.model.predict_proba(flat_img)[0]
        predicted_index = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_index]
        confidence = probabilities[predicted_index] * 100

        # Get additional information
        days_estimate = self.estimate_days_since_fallen(predicted_class, confidence)
        fall_reason = self.get_fall_reason(predicted_class, confidence)

        return {
            'class': predicted_class,
            'confidence': confidence,
            'days_estimate': days_estimate,
            'fall_reason': fall_reason,
            'probabilities': {
                class_name: float(prob * 100)
                for class_name, prob in zip(self.class_names, probabilities)
            }
        }