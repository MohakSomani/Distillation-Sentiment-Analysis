import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertForSequenceClassification
import matplotlib
matplotlib.use('Agg') 
def visualize_attention(text, model_path="models/transformers/bert-base-uncased"):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs, output_attentions=True)
    
    # Get attention weights from last layer
    attention = outputs.attentions[-1][0].detach().numpy().mean(axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(attention, cmap="viridis")
    plt.title("Attention Weights Visualization")
    plt.xlabel("Input Tokens")
    plt.ylabel("Input Tokens")
    # use fast tokenizer tokens() for labels
    plt.xticks(range(len(inputs.tokens())), inputs.tokens(), rotation=90)
    plt.yticks(range(len(inputs.tokens())), inputs.tokens())
    plt.savefig("logs/attention_weights.png")  # Save to file
    plt.close()  # Prevent memory leaks

if __name__ == "__main__":
    visualize_attention("I loved this movie despite its flaws")