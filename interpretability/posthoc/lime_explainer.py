import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

class LIMESentimentExplainer:
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.explainer = LimeTextExplainer(class_names=["negative", "positive"])

    def predict_proba(self, texts):
        return self.model.predict_proba(self.vectorizer.transform(texts))

    def explain(self, text):
        exp = self.explainer.explain_instance(text, self.predict_proba, num_features=10)
        return exp.as_html()

if __name__ == "__main__":
    explainer = LIMESentimentExplainer(
        "models/baselines/sst2_lr.pkl",
        "models/baselines/sst2_vectorizer.pkl"
    )
    print(explainer.explain("This movie was terrible!"))