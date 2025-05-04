import shap
from shap.maskers import Text as ShapTextMasker
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

class SHAPSentimentExplainer:
    def __init__(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        masker = ShapTextMasker(self.tokenizer)
        self.explainer = shap.Explainer(self.predict, masker)
    
    def predict(self, texts):
        # Handle SHAP inputs generically without class checks
        clean_texts = [str(t) for t in texts]
        inputs = self.tokenizer(clean_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=-1).numpy()
    
    def plot_explanation(self, text):
        shap_values = self.explainer([text])
        shap.plots.text(shap_values[:, :, 1], display=False)

if __name__ == "__main__":
    explainer = SHAPSentimentExplainer("models/transformers/bert-base-uncased")
    explainer.plot_explanation("The dialogue felt unnatural and forced")