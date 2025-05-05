import shap
from shap.maskers import Text as ShapTextMasker
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt

class SHAPSentimentExplainer:
    def __init__(self, model_path):
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        # Load model with eager attention to avoid SDPA warnings
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            attn_implementation="eager"  # Fixes attention warnings
        )
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        masker = ShapTextMasker(self.tokenizer)
        self.explainer = shap.Explainer(self.predict, masker)
    
    def predict(self, texts):
        clean_texts = [str(t) for t in texts]
        inputs = self.tokenizer(clean_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=-1).numpy()
    
    def plot_explanation(self, text):
        shap_values = self.explainer([text])
        # Save numeric SHAP values to txt for cross-checking
        # shap_values[0].feature_names holds tokens and shap_values[0].values[:,1] holds positive-class contributions
        tokens = shap_values[0].feature_names
        values = shap_values[0].values[:, 1]
        with open("logs/shap_values.txt", "w") as f:
            for tok, val in zip(tokens, values):
                f.write(f"{tok}\t{val}\n")
        # Save the plot to an HTML file instead of suppressing it
        shap_html = shap.plots.text(shap_values[:, :, 1], display=False)
        with open("logs/shap_results.html", "w") as f:
            f.write(shap_html)

if __name__ == "__main__":
    explainer = SHAPSentimentExplainer("models/transformers/bert-base-uncased")
    explainer.plot_explanation("The dialogue felt unnatural and forced")