import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datasets import load_from_disk
import joblib

def main(dataset_name):
    # Load dataset
    dataset = load_from_disk(f"data/raw/{dataset_name}")
    # select the train split for baseline training (DatasetDict or Dataset)
    try:
        split = dataset['train']
    except Exception:
        split = dataset
    # pick the appropriate text column
    if 'sentence' in split.column_names:
        text_col = 'sentence'
    else:
        text_col = 'text'
    texts = split[text_col]
    labels = split['label']

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
    
    # Train models
    lr = LogisticRegression().fit(X_train, y_train)
    svm = SVC(kernel='linear').fit(X_train, y_train)
    
    # Save models
    joblib.dump(vectorizer, f"models/baselines/{dataset_name}_vectorizer.pkl")
    joblib.dump(lr, f"models/baselines/{dataset_name}_lr.pkl")
    joblib.dump(svm, f"models/baselines/{dataset_name}_svm.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)