# Sentiment Intelligence Platform — E-Commerce Product Reviews
---

## Overview

Amazon star ratings are often misleading. A product might get 3 stars because shipping was late, not because the product is bad. This project builds a pipeline that goes beyond star ratings to extract *what specifically* customers like or dislike at an aspect level — battery life, build quality, price, etc. — and uses that intelligence to power a smarter, explainable recommendation system.

---

## Project Structure

```
sentiment_pipeline/
│
├── main.py                  ← Entry point — run this
├── config.py                ← All settings and hyperparameters
├── preprocess.py            ← Data loading, cleaning, label construction, splits
├── stage1_classical.py      ← TF-IDF + Logistic Regression & XGBoost
├── stage2_transformers.py   ← Zero-shot RoBERTa + Fine-tuned distilBERT
├── utils.py                 ← Shared helpers (evaluate, log_mlflow)
├── requirements.txt         ← Python dependencies
└── README.md
```

---

## Pipeline Stages

### Stage 1 — Classical ML Baselines
Establishes benchmark performance using traditional NLP approaches before applying deep learning.

| Model | Description |
|---|---|
| TF-IDF + Logistic Regression | Fast linear baseline with inverse-frequency class weights |
| TF-IDF + XGBoost | Gradient-boosted trees with per-sample class weights |

### Stage 2 — Transformer Models
Tests how much deep contextual language understanding improves over the classical baseline.

| Model | Description |
|---|---|
| Zero-shot RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment` with no fine-tuning |
| Fine-tuned distilBERT | `distilbert-base-uncased` fine-tuned on the review dataset |

### Stage 3 — Aspect-Based Sentiment Analysis *(coming soon)*
Extracts what customers are actually talking about — battery life, build quality, price — not just whether they're positive or negative.

### Stage 4 — Recommendation Engine *(coming soon)*
FAISS-based similarity search and gap-fill recommender over aspect sentiment vectors.

### Stage 5 — Agentic LLM Layer *(coming soon)*
LangChain AgentExecutor with ReAct reasoning and natural language explanations.

---

## Dataset

**Amazon Product Reviews** — Kaggle  
https://www.kaggle.com/datasets/manasabollavarapu/amazon-product-reviews/data

| Field | Description |
|---|---|
| `Text` | Full review text — primary NLP input |
| `Score` | Star rating (1–5) — used to construct sentiment labels |
| `ProductId` | Unique product identifier |
| `UserId` | Unique user identifier |
| `Summary` | Short review headline |

**Label Construction:**
```
1–2 stars  →  0  (negative)
3 stars    →  1  (neutral)
4–5 stars  →  2  (positive)
```

---

## Setup

### Requirements
- Python 3.10+
- Mac users: `brew install libomp` (required for XGBoost)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd sentiment_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup
1. Download the CSV from Kaggle
2. Place it in the project folder
3. Update `config.py`:
```python
DATA_PATH  = "your_file_name.csv"
TEXT_COL   = "Text"     # exact column name in your CSV
RATING_COL = "Score"    # exact column name in your CSV
```

---

## Running the Pipeline

```bash
# Activate venv
source venv/bin/activate

# Run full pipeline
python main.py
```

### What happens when you run it:
```
1. Preprocessing      → loads CSV, cleans text, builds labels, splits data
2. Stage 1-A          → trains TF-IDF + Logistic Regression
3. Stage 1-B          → trains TF-IDF + XGBoost
4. Stage 2-A          → runs zero-shot RoBERTa on test set
5. Stage 2-B          → fine-tunes distilBERT (this takes the longest)
6. Summary            → prints comparison table, saves results to JSON
```

### Expected runtime (Mac CPU, 50k samples):
| Stage | Approximate Time |
|---|---|
| Preprocessing | ~30 seconds |
| Logistic Regression | ~1–2 minutes |
| XGBoost | ~2–3 minutes |
| Zero-shot RoBERTa | ~2–3 minutes |
| Fine-tuned distilBERT | ~6–8 hours |

> **Tip:** To do a quick test run, set `SAMPLE_SIZE = 5_000` in `config.py`. distilBERT will finish in ~15–20 minutes.

---

## Configuration

All settings are in `config.py`. Key options:

```python
DATA_PATH   = "amazon_product_reviews.csv"  # path to your CSV
SAMPLE_SIZE = 50_000    # set to None for full dataset
BATCH_SIZE  = 32        # reduce to 16 if running out of memory
NUM_EPOCHS  = 4         # number of training epochs for distilBERT
```

---

## Viewing Results

### Terminal output
After all stages complete, a summary table is printed:
```
═════════════════════════════════════════════════════════
  Model                             Acc      Macro F1
═════════════════════════════════════════════════════════
  TF-IDF + LogReg               0.7668      0.7669
  TF-IDF + XGBoost              0.7812      0.7801
  Zero-shot RoBERTa             0.6543      0.6421
  Fine-tuned distilBERT         0.8321      0.8310
═════════════════════════════════════════════════════════
🏆  Best: Fine-tuned distilBERT  (macro_f1 = 0.8310)
```

### MLflow UI
```bash
# In a separate terminal window from the project folder
mlflow ui --port 5001
```
Open **http://localhost:5001** to see all runs, metrics, and saved model artifacts.

### Results JSON
`models/pipeline_results.json` — saved automatically after every run.

---

## Saved Models

| Model | Location |
|---|---|
| Logistic Regression | `models/logreg_model.pkl` |
| XGBoost | `models/xgboost_model.pkl` |
| Fine-tuned distilBERT | `models/distilbert/best_model/` |

> Note: Both the model and the vectorizer are saved together in the pickle files. You need both to make predictions on new reviews.

---

## Evaluation Metrics

| Metric | Why we use it |
|---|---|
| **Macro F1** | Primary metric — averages F1 equally across all 3 classes, accounts for class imbalance |
| **Accuracy** | Secondary metric — can be misleading due to positive-label skew in Amazon reviews |

---

## Key Design Decisions

- **Class weighting** — Amazon reviews are heavily positive-skewed (~70% 4–5 star). Both classical and transformer models use inverse-frequency class weights to prevent the model from just predicting positive all the time.
- **TF-IDF vocabulary fitted on training data only** — prevents data leakage from val/test sets.
- **Early stopping (patience=2)** — stops distilBERT training if validation macro F1 doesn't improve for 2 consecutive epochs, saving compute and preventing overfitting.
- **Stratified splits** — train/val/test splits preserve the class distribution so evaluation is fair.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, scikit-learn |
| Classical ML | TF-IDF, Logistic Regression, XGBoost |
| Deep learning | HuggingFace Transformers, distilBERT, RoBERTa |
| Experiment tracking | MLflow |
| Coming soon | spaCy, PyABSA, FAISS, ChromaDB, LangChain, FastAPI, Streamlit |