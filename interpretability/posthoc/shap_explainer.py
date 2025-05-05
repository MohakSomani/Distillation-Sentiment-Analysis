import shap
from shap.maskers import Text as ShapTextMasker
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import os
from matplotlib.colors import LinearSegmentedColormap

class SHAPSentimentExplainer:
    def __init__(self, model_path):
        import warnings
        warnings.filterwarnings("ignore")
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            attn_implementation="eager"
        )
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        masker = ShapTextMasker(self.tokenizer)
        self.explainer = shap.Explainer(self.predict, masker)
        
        # Custom colormap (red for negative, blue for positive)
        self.cmap = LinearSegmentedColormap.from_list("shap", ["red", "white", "blue"])
    
    def predict(self, texts):
        clean_texts = [str(t) for t in texts]
        inputs = self.tokenizer(clean_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=-1).numpy()
    
    def plot_explanation(self, text):
        # Get SHAP values
        shap_values = self.explainer([text])
        
        # Save raw values
        self._save_raw_values(shap_values)
        
        # Generate enhanced HTML visualization
        self._generate_html_visualization(shap_values, text)
        
        # Generate matplotlib visualization
        self._generate_matplotlib_visualization(shap_values, text)
    
    def _save_raw_values(self, shap_values):
        """Save raw SHAP values to a text file"""
        with open("logs/shap_values.txt", "w") as f:
            for token, value in zip(shap_values[0].feature_names, shap_values[0].values[:, 1]):
                f.write(f"{token}\t{value:.6f}\n")
    
    def _generate_html_visualization(self, shap_values, text):
        """Generate enhanced HTML visualization"""
        # Generate original HTML
        html = shap.plots.text(shap_values[:, :, 1], display=False)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Create new container div
        container = soup.new_tag("div", style=(
            "display: flex; flex-direction: column; gap: 4px; "
            "max-width: 800px; margin: 0 auto; padding: 20px; "
            "font-family: Arial, sans-serif;"
        ))
        
        # Add title
        title_div = soup.new_tag("div", style=(
            "font-size: 18px; font-weight: bold; "
            "margin-bottom: 15px; text-align: center;"
        ))
        title_div.string = "SHAP Explanation for:"
        container.append(title_div)
        
        # Add original text
        text_div = soup.new_tag("div", style=(
            "background: #f8f8f8; padding: 10px; "
            "border-radius: 5px; margin-bottom: 15px;"
        ))
        text_div.string = text
        container.append(text_div)
        
        # Process each token span
        for span in soup.find_all('span'):
            if not span.text.strip():
                continue
                
            # Create new row for each token
            token_div = soup.new_tag("div", style=(
                "display: flex; align-items: center; "
                "justify-content: space-between; "
                "padding: 5px 10px;"
            ))
            
            # Get SHAP value
            value = float(span.get('title', '0').split('=')[-1].strip())
            abs_value = abs(value)
            
            # Create colored value indicator
            value_span = soup.new_tag("span", style=(
                f"width: 100px; font-family: monospace; "
                f"font-weight: bold; text-align: right; "
                f"color: {'#d62728' if value < 0 else '#1f77b4'};"
            ))
            value_span.string = f"{value:+.4f}"
            
            # Create token display with color intensity based on impact
            intensity = min(abs_value * 10, 1.0)
            bg_color = f"rgba(214, 39, 40, {intensity})" if value < 0 else f"rgba(31, 119, 180, {intensity})"
            
            token_span = soup.new_tag("span", style=(
                f"padding: 4px 8px; margin-left: 10px; "
                f"background: {bg_color}; color: white; "
                f"border-radius: 4px; flex-grow: 1;"
            ))
            token_span.string = span.text
            
            # Add elements to row
            token_div.append(token_span)
            token_div.append(value_span)
            container.append(token_div)
        
        # Replace original content
        center_div = soup.find('div', {'align': 'center'})
        if center_div:
            center_div.clear()
            center_div.append(container)
        
        # Add footer
        footer = soup.new_tag("div", style=(
            "margin-top: 20px; font-size: 12px; "
            "color: #777; text-align: center;"
        ))
        footer.string = "Blue = Positive impact, Red = Negative impact"
        container.append(footer)
        
        # Save enhanced HTML
        with open("logs/shap_results.html", "w") as f:
            f.write(str(soup.prettify()))
    
    def _generate_matplotlib_visualization(self, shap_values, text):
        """Generate matplotlib-based visualization"""
        # Get tokens and values
        tokens = shap_values[0].feature_names
        values = shap_values[0].values[:, 1]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar colors based on values
        colors = [self.cmap(0.5 + 0.5 * v / max(abs(values))) for v in values]
        
        # Plot bars
        bars = plt.barh(tokens, values, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x = width if width >= 0 else width
            plt.text(label_x, bar.get_y() + bar.get_height()/2,
                     f"{width:+.4f}",
                     ha='left' if width >= 0 else 'right',
                     va='center',
                     color='white' if abs(width) > 0.1 else 'black',
                     fontweight='bold')
        
        # Style the plot
        plt.title(f"SHAP Explanation for: {textwrap.shorten(text, width=60)}", pad=20)
        plt.xlabel("SHAP Value (Impact on Model Output)")
        plt.ylabel("Token")
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig("logs/shap_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved visualization to logs/shap_plot.png")

if __name__ == "__main__":
    explainer = SHAPSentimentExplainer("models/transformers/bert-base-uncased")
    explainer.plot_explanation("The movie was absolutely fantastic! I loved every moment of it.")