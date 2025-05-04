# Sentiment Analysis with Interpretability

## Setup
```
pip install -r requirements.txt
bash data/download_datasets.sh
```
## Training Models

Combined script to train all in 
scripts/train_all_models.sh

### Train SVM/Logistic Regression
```
python models/baselines/train_baselines.py --dataset sst2
```
### Train BERT
```
python models/transformers/train_bert.py
```
## Generating Explanations

Combined script to run all interpretations in 
scripts/run_interpretability.sh

## LIME Explanations
```
python interpretability/posthoc/lime_explainer.py
```

## Attention Visualization
```
python interpretability/self_interpret/attention_viz.py
```

## Results
See docs/Final_Report.pdf for performance comparisons.

