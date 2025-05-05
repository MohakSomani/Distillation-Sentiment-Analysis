import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
def plot_feature_importance(model_path, vectorizer_path, top_n=20):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    features = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]  # For logistic regression
    
    # Get top N features
    top_indices = np.argsort(np.abs(coefs))[-top_n:]
    top_features = [features[i] for i in top_indices]
    top_coefs = coefs[top_indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['green' if c > 0 else 'red' for c in top_coefs]
    plt.barh(top_features, top_coefs, color=colors)
    plt.title("Top {} Important Features".format(top_n))
    plt.savefig("logs/feature_importance.png")  # Save to file
    # Save numeric values to txt for cross-checking
    with open("logs/feature_importance.txt", "w") as f:
        for feature, coef in zip(top_features, top_coefs):
            f.write(f"{feature}\t{coef}\n")
    plt.close()  # Prevent memory leaks

if __name__ == "__main__":
    plot_feature_importance(
        "models/baselines/sst2_lr.pkl",
        "models/baselines/sst2_vectorizer.pkl"
    )