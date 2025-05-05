# 📊 Sentiment Analysis with Interpretability

A comparison of traditional (SVM, Logistic Regression) and transformer-based (BERT, DistilBERT) sentiment models, enhanced with both post-hoc (LIME, SHAP) and self-interpretation (attention, feature importance) methods.

---

## 📋 Table of Contents

1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Getting Started](#getting-started)  
   1. [Prerequisites](#prerequisites)  
   2. [Installation & Data](#installation--data)  
4. [Usage](#usage)  
   1. [Training Models](#training-models)  
   2. [Generating Explanations](#generating-explanations)  
5. [Configuration](#configuration)  
6. [Notebooks & Demos](#notebooks--demos)  
7. [Tests](#tests)  
8. [Results](#results)  

---

## 🌟 Features

- Traditional ML: TF-IDF + Logistic Regression & SVM  
- Transformer Models: BERT fine-tuning & knowledge distillation to DistilBERT  
- Post-hoc explainers: LIME, SHAP  
- Self-interpretation: attention visualizations, feature importance  
- Automated scripts for end-to-end workflows  

---

## 🗂️ Project Structure

```
Distillation-Sentiment-Analysis/
├── configs/                 # YAML configs for baselines & BERT
├── data/                    # Raw & processed datasets
├── docs/                    # Final_Report.pdf, slides, etc.
├── interpretability/        # LIME, SHAP, attention & feature scripts
├── models/
│   ├── baselines/           # TF-IDF + SVM/LR training code
│   └── transformers/        # BERT fine-tune & distillation
├── notebooks/               # EDA & interactive demos
├── scripts/                 # Automations: train_all_models.sh, run_interpretability.sh
├── tests/                   # Unit tests for data loading & models
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- `pip`

### Installation & Data

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
bash data/download_datasets.sh
```

---

## ⚙️ Usage

### Training Models

#### 1. Train Baselines (SVM / Logistic Regression)

```bash
python models/baselines/train_baselines.py --dataset sst2
```

#### 2. Train BERT

```bash
python models/transformers/train_bert.py
```

#### 3. Distill to DistilBERT

```bash
python models/transformers/distillation.py
```

*Or run all at once*:

```bash
bash scripts/train_all_models.sh
```

---

### Generating Explanations

#### 1. LIME (post-hoc)

```bash
python interpretability/posthoc/lime_explainer.py
```

#### 2. SHAP (post-hoc)

```bash
python interpretability/posthoc/shap_explainer.py
```

#### 3. Attention Visualization (self-interpret)

```bash
python interpretability/self_interpret/attention_viz.py
```

#### 4. Feature Importance (self-interpret)

```bash
python interpretability/self_interpret/feature_importance.py
```

*Or run all explanations at once*:

```bash
bash scripts/run_interpretability.sh
```

---

## 🛠️ Configuration

- `configs/baseline_config.yaml`: TF-IDF & baseline model hyperparameters  
- `configs/bert_config.yaml`: BERT fine-tuning hyperparameters  

---

## 📓 Notebooks & Demos

- `notebooks/EDA.ipynb`: Exploratory Data Analysis  
- `notebooks/Interpretation_Demo.ipynb`: Interactive LIME, SHAP & attention demos  

---

## ✅ Tests

```bash
pytest --maxfail=1 --disable-warnings -q
```

- `tests/test_data_loading.py`  
- `tests/test_models.py`  

---

## 📈 Results

- See `docs/Final_Report.pdf` for accuracy/F1 comparisons.  
- Generated plots in `logs/`:  
  - `attention_weights.png`  
  - `feature_importance.png`  
- LIME/SHAP outputs also in `logs/`.  

---

© 2025 Sentiment Analysis Team.

