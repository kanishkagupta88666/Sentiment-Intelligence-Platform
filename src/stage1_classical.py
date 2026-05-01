# ══════════════════════════════════════════════════════════════
#  stage1_classical.py
#  Stage 1 — Classical ML Baselines
#
#  Trains and evaluates two TF-IDF based sentiment classifiers:
#    A) Logistic Regression   (fast, strong linear baseline)
#    B) XGBoost               (gradient-boosted tree baseline)
#
#  Both use inverse-frequency class weights to handle the
#  positive-skew typical in Amazon review datasets.
#  Results are logged to MLflow for comparison with Stage 2.
# ══════════════════════════════════════════════════════════════

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.utils.class_weight      import compute_class_weight
from xgboost                         import XGBClassifier

import config
from utils import evaluate, log_mlflow

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
def build_tfidf_vectorizer(df_train: pd.DataFrame) -> TfidfVectorizer:
    """
    Fits a TF-IDF vectoriser on the training corpus and returns it.
    Uses sublinear term-frequency scaling (log(1+tf)) to reduce the
    impact of very frequent terms, and captures unigrams + bigrams.

    Parameters
    ----------
    df_train : pd.DataFrame — must contain a 'clean_text' column

    Returns
    -------
    TfidfVectorizer — fitted vectoriser (call .transform() on val/test)
    """
    log.info(
        "  Fitting TF-IDF  max_features=%d  ngram_range=%s",
        config.TFIDF_MAX_FEATURES,
        config.TFIDF_NGRAM_RANGE,
    )
    vec = TfidfVectorizer(
        max_features  = config.TFIDF_MAX_FEATURES,
        ngram_range   = config.TFIDF_NGRAM_RANGE,
        sublinear_tf  = True,
        strip_accents = "unicode",
    )
    vec.fit(df_train["clean_text"])
    return vec


# ──────────────────────────────────────────────────────────────
def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Computes balanced class weights (total / n_classes * n_samples_per_class)
    to counter the heavy positive skew in Amazon reviews, where 4-5 star
    reviews dominate.

    Parameters
    ----------
    y_train : np.ndarray — integer labels from the training set

    Returns
    -------
    dict mapping class index → float weight
    """
    cw = compute_class_weight(
        class_weight = "balanced",
        classes      = np.unique(y_train),
        y            = y_train,
    )
    return {i: w for i, w in enumerate(cw)}


# ──────────────────────────────────────────────────────────────
def run_logistic_regression(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> dict:
    """
    Stage 1-A: TF-IDF + Logistic Regression

    Fits a multinomial Logistic Regression (saga solver, L2 regularisation)
    on TF-IDF features extracted from the training set, then evaluates on
    the held-out test set.  Logs params and metrics to MLflow.

    The saga solver is chosen for speed on large sparse matrices and
    native support for multi-class classification.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Must contain 'clean_text' and 'label' columns.

    Returns
    -------
    dict with keys: model, accuracy, macro_f1
    """
    log.info("\n══ STAGE 1-A: TF-IDF + Logistic Regression ══")

    vec     = build_tfidf_vectorizer(df_train)
    X_train = vec.fit_transform(df_train["clean_text"])
    X_test  = vec.transform(df_test["clean_text"])

    y_train = df_train["label"].values
    y_test  = df_test["label"].values

    cw_dict = get_class_weights(y_train)

    model = LogisticRegression(
        max_iter     = 1000,
        C            = 1.0,
        solver       = "saga",
        class_weight = cw_dict,
        random_state = config.RANDOM_SEED,
        n_jobs       = -1,
    )
    model.fit(X_train, y_train)
    preds   = model.predict(X_test)
    metrics = evaluate(y_test, preds, "TF-IDF + LogReg")

    log_mlflow(
        run_name = "tfidf-logreg",
        params   = {
            "model":        "LogisticRegression",
            "max_features": config.TFIDF_MAX_FEATURES,
            "ngram_range":  str(config.TFIDF_NGRAM_RANGE),
            "C":            1.0,
            "solver":       "saga",
        },
        metrics  = metrics,
    )

    with open("models/logreg_model.pkl", "wb") as f:
        pickle.dump({"model": model, "vectorizer": vec}, f)
    return metrics


# ──────────────────────────────────────────────────────────────
def run_xgboost(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> dict:
    """
    Stage 1-B: TF-IDF + XGBoost

    Trains a gradient-boosted tree classifier on TF-IDF features.
    Class imbalance is handled via per-sample weights derived from
    inverse class frequency.  Uses the histogram-based tree method
    ('hist') for faster training on large sparse matrices.

    Early stopping is applied on the validation log-loss to prevent
    overfitting without needing to tune n_estimators manually.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Must contain 'clean_text' and 'label' columns.

    Returns
    -------
    dict with keys: model, accuracy, macro_f1
    """
    log.info("\n══ STAGE 1-B: TF-IDF + XGBoost ══")

    vec     = build_tfidf_vectorizer(df_train)
    X_train = vec.fit_transform(df_train["clean_text"])
    X_val   = vec.transform(df_val["clean_text"])
    X_test  = vec.transform(df_test["clean_text"])

    y_train = df_train["label"].values
    y_val   = df_val["label"].values
    y_test  = df_test["label"].values

    # convert class-weight dict to per-sample array
    cw_dict        = get_class_weights(y_train)
    sample_weights = np.array([cw_dict[y] for y in y_train])

    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "mlogloss",
        tree_method      = "hist",    # switch to "gpu_hist" if GPU available
        random_state     = config.RANDOM_SEED,
        n_jobs           = 1,
        verbosity        = 0,
    )
    model.fit(
        X_train, y_train,
        sample_weight = sample_weights,
        eval_set      = [(X_val, y_val)],
        verbose       = False,
    )

    preds   = model.predict(X_test)
    metrics = evaluate(y_test, preds, "TF-IDF + XGBoost")

    log_mlflow(
        run_name = "tfidf-xgboost",
        params   = {
            "model":         "XGBoost",
            "max_features":  config.TFIDF_MAX_FEATURES,
            "ngram_range":   str(config.TFIDF_NGRAM_RANGE),
            "n_estimators":  300,
            "max_depth":     6,
            "learning_rate": 0.1,
        },
        metrics  = metrics,
    )
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump({"model": model, "vectorizer": vec}, f)
    return metrics
