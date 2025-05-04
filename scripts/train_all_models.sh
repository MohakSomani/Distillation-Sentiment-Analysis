#!/bin/bash

# Train baselines
python3 models/baselines/train_baselines.py --dataset sst2
python3 models/baselines/train_baselines.py --dataset imdb

# Train BERT
python3 models/transformers/train_bert.py