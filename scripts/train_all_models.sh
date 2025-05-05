#!/bin/bash

# Train baselines
echo "Training baselines on SST-2 dataset..."
python3 models/baselines/train_baselines.py --dataset sst2

echo "Training baselines on IMDB dataset..."
python3 models/baselines/train_baselines.py --dataset imdb

# Train BERT
echo "Training BERT model..."
python3 models/transformers/train_bert.py

# Train DistilBERT via distillation
echo "Training DistilBERT model via distillation..."
python3 models/transformers/distillation.py