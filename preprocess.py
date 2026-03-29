# ══════════════════════════════════════════════════════════════
#  preprocess.py
#  Loads the raw Kaggle Amazon reviews CSV, cleans the text,
#  constructs 3-class sentiment labels from star ratings,
#  and produces stratified train / val / test splits.
# ══════════════════════════════════════════════════════════════

import re
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

import config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Normalises a raw review string by:
      - lowercasing everything
      - stripping HTML tags and URLs
      - removing non-alphanumeric characters (keeps basic punctuation)
      - collapsing repeated whitespace

    Parameters
    ----------
    text : str — raw review text

    Returns
    -------
    str — cleaned text ready for vectorisation / tokenisation
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>",   " ", text)           # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)       # URLs
    text = re.sub(r"[^a-z0-9\s,.!?']", " ", text)     # special characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────
def rating_to_label(rating) -> int:
    """
    Converts a 1-5 star rating into a 3-class sentiment label:
      1–2  →  0  (negative)
      3    →  1  (neutral)
      4–5  →  2  (positive)

    Parameters
    ----------
    rating : int or str — raw star rating from the dataset

    Returns
    -------
    int — class label (0, 1, or 2)
    """
    r = int(rating)
    if r <= 2:
        return 0   # negative
    if r == 3:
        return 1   # neutral
    return 2       # positive


# ──────────────────────────────────────────────────────────────
def load_and_preprocess():
    """
    Full preprocessing pipeline:
      1. Reads the CSV from config.DATA_PATH
      2. Drops rows with missing text or out-of-range ratings
      3. Removes very short reviews (< 5 words) — too sparse for ABSA
      4. Applies clean_text() to the review body
      5. Constructs the 3-class 'label' column via rating_to_label()
      6. Optionally down-samples to config.SAMPLE_SIZE (stratified)
      7. Produces 80 / 10 / 10 train / val / test splits (stratified)

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
        Each DataFrame contains at minimum:
          - 'clean_text' : preprocessed review string
          - 'label'      : integer class (0 / 1 / 2)
    """
    log.info("Loading CSV: %s", config.DATA_PATH)
    df = pd.read_csv(config.DATA_PATH)
    log.info("  Raw rows loaded: %d", len(df))

    # ── keep only needed columns ──────────────────────────────
    df = df[[config.TEXT_COL, config.RATING_COL]].copy()
    df.dropna(inplace=True)

    # ── remove invalid ratings ────────────────────────────────
    df = df[df[config.RATING_COL].astype(str).str.match(r"^[1-5]$")]
    df[config.RATING_COL] = df[config.RATING_COL].astype(int)

    # ── remove very short reviews ─────────────────────────────
    df = df[df[config.TEXT_COL].str.split().str.len() >= 5]
    log.info("  After cleaning: %d rows", len(df))

    # ── build sentiment label ─────────────────────────────────
    df["label"]      = df[config.RATING_COL].apply(rating_to_label)
    df["clean_text"] = df[config.TEXT_COL].apply(clean_text)

    # ── optional stratified sampling ─────────────────────────
    if config.SAMPLE_SIZE and config.SAMPLE_SIZE < len(df):
        df = (
            df.groupby("label", group_keys=False)
            .apply(
                lambda g: g.sample(
                    min(len(g), config.SAMPLE_SIZE // 3),
                    random_state=config.RANDOM_SEED,
                )
            )
            .reset_index(drop=True)
        )
        log.info("  Sampled to %d rows (stratified by label)", len(df))

    # ── train / val / test split ─────────────────────────────
    df_train, df_temp = train_test_split(
        df,
        test_size    = config.VAL_SIZE + config.TEST_SIZE,
        stratify     = df["label"],
        random_state = config.RANDOM_SEED,
    )
    relative_test = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    df_val, df_test = train_test_split(
        df_temp,
        test_size    = relative_test,
        stratify     = df_temp["label"],
        random_state = config.RANDOM_SEED,
    )

    log.info(
        "  Splits → train=%d  val=%d  test=%d",
        len(df_train), len(df_val), len(df_test),
    )

    # ── log label distribution per split ─────────────────────
    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dist = split["label"].value_counts(normalize=True).sort_index()
        log.info(
            "  %s dist: %s",
            name,
            {config.LABEL_NAMES[i]: f"{v:.1%}" for i, v in dist.items()},
        )

    return df_train, df_val, df_test
