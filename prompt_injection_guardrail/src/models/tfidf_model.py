import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os

# Get repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class TfidfClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(REPO_ROOT, "models/tfidf_model.pkl")
        self.model_path = model_path
        self.pipeline = None

    def train(self, data_path=None):
        if data_path is None:
            data_path = os.path.join(REPO_ROOT, "data/processed/dataset.csv")
            
        print(f"Training Tier 1 (TF-IDF + Logistic Regression) from {data_path}...")
        df = pd.read_csv(data_path)
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb', 
                ngram_range=(3, 5), 
                max_features=50000
            )),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        self.pipeline.fit(df['text'], df['label'])
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved to {self.model_path}")

    def predict(self, text):
        if self.pipeline is None:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.pipeline = pickle.load(f)
            else:
                raise Exception(f"Model not found at {self.model_path}. Please train it first.")
        
        probs = self.pipeline.predict_proba([text])[0]
        label = self.pipeline.predict([text])[0]
        confidence = max(probs)
        return label, confidence

if __name__ == "__main__":
    clf = TfidfClassifier()
    clf.train()
