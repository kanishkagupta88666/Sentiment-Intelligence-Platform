# ══════════════════════════════════════════════════════════════
#  config.py
#  Central configuration for the entire sentiment pipeline.
#  Edit values here — nothing else needs to change.
# ══════════════════════════════════════════════════════════════

import torch

# ── Dataset ───────────────────────────────────────────────────
DATA_PATH  = "/Users/shithilshetty/Documents/Projects/sentiment analysis/Product Reviews.csv"   # path to the Kaggle CSV
TEXT_COL   = "Text"                  # free-text review column
RATING_COL = "Score"                       # 1-5 star rating column

# ── Sampling ──────────────────────────────────────────────────
# Set to None to train on the full 500k+ dataset.
# Use 50_000 for quick local iteration.
SAMPLE_SIZE = 50_000

# ── Train / Val / Test splits ─────────────────────────────────
VAL_SIZE    = 0.10
TEST_SIZE   = 0.10
RANDOM_SEED = 42

# ── TF-IDF (Stage 1) ──────────────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)    # unigrams + bigrams

# ── Transformer model checkpoints (Stage 2) ───────────────────
ZERO_SHOT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
DISTILBERT_CKPT = "distilbert-base-uncased"

# ── Transformer training hyperparameters ─────────────────────
MAX_LENGTH   = 256
BATCH_SIZE   = 32
NUM_EPOCHS   = 4
LR           = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
FP16         = torch.cuda.is_available()   # auto-enable on GPU

# ── Output & experiment tracking ─────────────────────────────
OUTPUT_DIR   = "models"
MLFLOW_URI   = "mlruns"
MLFLOW_EXP   = "sentiment-pipeline"

# ── Label definitions (do not change order) ──────────────────
LABEL_NAMES = ["negative", "neutral", "positive"]   # indices 0 / 1 / 2
NUM_LABELS  = len(LABEL_NAMES)
