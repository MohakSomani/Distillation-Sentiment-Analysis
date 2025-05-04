#!/bin/bash

# Generate all explanations
mkdir -p logs
python3 interpretability/posthoc/lime_explainer.py > logs/lime_results.html
python3 interpretability/posthoc/shap_explainer.py > logs/shap_results.html
python3 interpretability/self_interpret/feature_importance.py
python3 interpretability/self_interpret/attention_viz.py