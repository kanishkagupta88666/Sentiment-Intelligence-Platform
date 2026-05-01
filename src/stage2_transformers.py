# ══════════════════════════════════════════════════════════════
#  stage2_transformers.py
#  Stage 2 — Transformer Models
#
#  A) Zero-shot evaluation using cardiffnlp/twitter-roberta-base-sentiment
#     No training involved — tests how well an off-the-shelf model
#     generalises to Amazon product review sentiment.
#
#  B) Fine-tuning distilbert-base-uncased on the labelled review
#     dataset using HuggingFace Trainer with:
#       - Weighted cross-entropy loss (handles class imbalance)
#       - Early stopping on validation macro F1
#       - Best model checkpoint saved to disk
#       - All runs logged to MLflow
# ══════════════════════════════════════════════════════════════

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    pipeline as hf_pipeline,
)

import config
from utils import evaluate, log_mlflow

log = logging.getLogger(__name__)

# Cardiff RoBERTa label map:
#   LABEL_0 → negative (0)
#   LABEL_1 → neutral  (1)
#   LABEL_2 → positive (2)
CARDIFF_LABEL_MAP = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}


# ──────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    """
    PyTorch Dataset that tokenises a DataFrame of reviews
    and stores input IDs, attention masks, and integer labels.
    Passed directly into the HuggingFace Trainer.

    Parameters
    ----------
    df        : pd.DataFrame — must contain 'clean_text' and 'label'
    tokenizer : PreTrainedTokenizer — HuggingFace tokenizer to apply
    max_length: int — sequences are padded / truncated to this length
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.encodings = tokenizer(
            df["clean_text"].tolist(),
            truncation     = True,
            padding        = "max_length",
            max_length     = max_length,
            return_tensors = "pt",
        )
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ──────────────────────────────────────────────────────────────
def compute_metrics_hf(eval_pred) -> dict:
    """
    Callback used by HuggingFace Trainer to compute evaluation metrics
    at the end of each epoch.  Returns accuracy and macro F1,
    which are printed to the training log and used for model selection.

    Parameters
    ----------
    eval_pred : EvalPrediction — namedtuple from Trainer
                (predictions=logits, label_ids=true labels)

    Returns
    -------
    dict with keys: accuracy, macro_f1
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# ──────────────────────────────────────────────────────────────
def build_weighted_trainer_class(class_weights: torch.Tensor):
    """
    Dynamically constructs a Trainer subclass that replaces the default
    cross-entropy loss with a weighted version.

    This is necessary because the standard HuggingFace Trainer does not
    expose a class_weight argument — we override compute_loss() instead.
    The weights are captured in the closure so the subclass remains
    compatible with all other Trainer functionality.

    Parameters
    ----------
    class_weights : torch.Tensor — shape (num_classes,), one weight per class

    Returns
    -------
    WeightedTrainer class (not yet instantiated)
    """

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(outputs.logits.device)
            )
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# ──────────────────────────────────────────────────────────────
def run_zero_shot(df_test: pd.DataFrame) -> dict:
    """
    Stage 2-A: Zero-shot sentiment classification with Cardiff RoBERTa.

    Runs cardiffnlp/twitter-roberta-base-sentiment directly on the test
    set with no fine-tuning.  Provides an upper-bound estimate for what
    a pre-trained model achieves on this domain out of the box, and
    sets the bar that fine-tuning must beat.

    Parameters
    ----------
    df_test : pd.DataFrame — must contain 'clean_text' and 'label'

    Returns
    -------
    dict with keys: model, accuracy, macro_f1
    """
    log.info("\n══ STAGE 2-A: Zero-shot RoBERTa ══")

    clf = hf_pipeline(
        "text-classification",
        model      = config.ZERO_SHOT_MODEL,
        tokenizer  = config.ZERO_SHOT_MODEL,
        truncation = True,
        max_length = config.MAX_LENGTH,
        device     = 0 if torch.cuda.is_available() else -1,
        batch_size = config.BATCH_SIZE,
    )

    texts  = df_test["clean_text"].tolist()
    labels = df_test["label"].values

    raw_preds = clf(texts)
    preds     = np.array([CARDIFF_LABEL_MAP[p["label"]] for p in raw_preds])

    metrics = evaluate(labels, preds, "Zero-shot RoBERTa")
    log_mlflow(
        run_name = "zero-shot-roberta",
        params   = {
            "model":      config.ZERO_SHOT_MODEL,
            "zero_shot":  True,
            "max_length": config.MAX_LENGTH,
        },
        metrics = metrics,
    )
    return metrics


# ──────────────────────────────────────────────────────────────
def run_distilbert(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> dict:
    """
    Stage 2-B: Fine-tune distilbert-base-uncased for 3-class sentiment.

    Training procedure:
      1. Tokenise all three splits with the distilBERT WordPiece tokeniser.
      2. Compute inverse-frequency class weights from the training set.
      3. Fine-tune with the WeightedTrainer (weighted cross-entropy loss).
      4. Monitor validation macro F1 each epoch; early-stop if no
         improvement for 2 consecutive epochs.
      5. Save the best checkpoint to disk and log it as an MLflow artifact.
      6. Evaluate the best checkpoint on the held-out test set.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Must contain 'clean_text' and 'label' columns.

    Returns
    -------
    dict with keys: model, accuracy, macro_f1
    """
    log.info("\n══ STAGE 2-B: Fine-tuned distilBERT ══")

    out_dir = Path(config.OUTPUT_DIR) / "distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── tokeniser & model ─────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.DISTILBERT_CKPT)
    model     = AutoModelForSequenceClassification.from_pretrained(
        config.DISTILBERT_CKPT,
        num_labels = config.NUM_LABELS,
        id2label   = {i: l for i, l in enumerate(config.LABEL_NAMES)},
        label2id   = {l: i for i, l in enumerate(config.LABEL_NAMES)},
    )

    # ── tokenised PyTorch datasets ────────────────────────────
    train_ds = ReviewDataset(df_train, tokenizer, config.MAX_LENGTH)
    val_ds   = ReviewDataset(df_val,   tokenizer, config.MAX_LENGTH)
    test_ds  = ReviewDataset(df_test,  tokenizer, config.MAX_LENGTH)

    # ── class weights to counter positive-label skew ─────────
    y_train = df_train["label"].values
    raw_cw  = compute_class_weight(
        class_weight = "balanced",
        classes      = np.unique(y_train),
        y            = y_train,
    )
    class_weights = torch.tensor(raw_cw, dtype=torch.float)
    log.info("  Class weights: %s", class_weights.tolist())

    # ── training arguments ────────────────────────────────────
    steps_per_epoch = max(1, len(train_ds) // config.BATCH_SIZE)
    training_args   = TrainingArguments(
        output_dir                  = str(out_dir),
        num_train_epochs            = config.NUM_EPOCHS,
        per_device_train_batch_size = config.BATCH_SIZE,
        per_device_eval_batch_size  = config.BATCH_SIZE,
        learning_rate               = config.LR,
        warmup_ratio                = config.WARMUP_RATIO,
        weight_decay                = config.WEIGHT_DECAY,
        fp16                        = config.FP16,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "macro_f1",
        greater_is_better           = True,
        logging_dir                 = str(out_dir / "logs"),
        logging_steps               = steps_per_epoch // 4,
        save_total_limit            = 2,
        report_to                   = "none",   # MLflow handled manually
        seed                        = config.RANDOM_SEED,
    )

    # ── build weighted trainer and train ──────────────────────
    WeightedTrainer = build_weighted_trainer_class(class_weights)
    trainer         = WeightedTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics_hf,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # ── evaluate on test set ──────────────────────────────────
    raw_output = trainer.predict(test_ds)
    preds      = np.argmax(raw_output.predictions, axis=-1)
    metrics    = evaluate(df_test["label"].values, preds, "Fine-tuned distilBERT")

    # ── save best model and tokeniser ────────────────────────
    best_dir = out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    log.info("  Best model saved → %s", best_dir)

    log_mlflow(
        run_name      = "distilbert-finetuned",
        params        = {
            "model":       config.DISTILBERT_CKPT,
            "max_length":  config.MAX_LENGTH,
            "batch_size":  config.BATCH_SIZE,
            "num_epochs":  config.NUM_EPOCHS,
            "lr":          config.LR,
        },
        metrics       = metrics,
        artifact_path = str(best_dir),
    )
    return metrics
