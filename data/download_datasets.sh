#!/bin/bash

# Create directories
mkdir -p data/raw/sst2 data/raw/imdb data/raw/ag_news

# Download SST-2
python3 -c "from datasets import load_dataset; load_dataset('sst2').save_to_disk('data/raw/sst2')"

# Download IMDB
python3 -c "from datasets import load_dataset; load_dataset('imdb').save_to_disk('data/raw/imdb')"
# python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'"  
# Download AG News (fixed command)
python3 -c "import tensorflow_datasets as tfds; ds = tfds.load(\"ag_news_subset\", split=\"train\"); tfds.as_dataframe(ds).to_csv(\"data/raw/ag_news/train.csv\")"