import unittest
import joblib
from transformers import BertForSequenceClassification, BertTokenizer

class TestModels(unittest.TestCase):
    def test_bert_inference(self):
        model = BertForSequenceClassification.from_pretrained("models/transformers/bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("models/transformers/bert-base-uncased")
        inputs = tokenizer("This is a test", return_tensors="pt")
        outputs = model(**inputs)
        self.assertEqual(outputs.logits.shape, (1, 2))

    def test_baseline_model(self):
        model = joblib.load("models/baselines/sst2_lr.pkl")
        vectorizer = joblib.load("models/baselines/sst2_vectorizer.pkl")
        pred = model.predict(vectorizer.transform(["positive text"]))
        self.assertIn(pred[0], [0, 1])

if __name__ == "__main__":
    unittest.main()