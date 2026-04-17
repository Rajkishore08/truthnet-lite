import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

class FakeNewsDetector:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        self.pipeline = None
        self.load_or_train_model()

    def load_or_train_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
        else:
            print(f"Warning/Error: Production Kaggle dataset model not found at {self.model_path}")
            self.pipeline = None

    def analyze(self, text: str):
        if not text.strip() or self.pipeline is None:
            return {"prediction": "Unknown"}
        
        prediction = self.pipeline.predict([text])[0]
        
        # PassiveAggressiveClassifier doesn't provide predict_proba directly, but we can use decision_function
        decision = 100
        if hasattr(self.pipeline.named_steps['clf'], 'decision_function'):
            dist = self.pipeline.decision_function([text])[0]
            # Convert abstract distance to a pseudo-confidence score
            import math
            confidence = 1 / (1 + math.exp(-abs(dist))) * 100
            decision = round(confidence, 2)
        
        return {
            "prediction": "Real News" if prediction == 1 else "Fake News",
            "confidence": decision
        }
