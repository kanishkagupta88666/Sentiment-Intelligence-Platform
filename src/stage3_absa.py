# ══════════════════════════════════════════════════════════════
#  stage3_absa.py
#  Stage 3 — Aspect-Based Sentiment Analysis (ABSA)
#
#  Task 1: Aspect Extraction using spaCy noun chunks
#  Task 2: Aspect-level Sentiment via zero-shot NLI classifier
#  Task 3: Clean & Filter results (remove generic, deduplicate)
#
#  Outputs are saved as intermediate files after each task so
#  you never have to re-run everything from scratch.
# ══════════════════════════════════════════════════════════════

import json
import logging
import re
from pathlib import Path

import pandas as pd
import spacy
import torch
from transformers import pipeline as hf_pipeline

import config

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(config.OUTPUT_DIR) / "absa"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  TASK 1 — Aspect Extraction (spaCy noun chunks)
# ──────────────────────────────────────────────────────────────
def extract_aspects_spacy(df: pd.DataFrame) -> dict:
    """
    Extracts candidate aspect phrases from each review using
    spaCy noun chunks.

    Parameters
    ----------
    df : pd.DataFrame — must have 'clean_text' column and index as review_id

    Returns
    -------
    dict : review_id → list of aspect strings
    """
    log.info("═══ TASK 1: Aspect Extraction (spaCy) ═══")

    # ── load existing results to resume ───────────────────────
    out_path = OUTPUT_DIR / "task1_aspects_spacy.json"
    aspects_per_review = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        aspects_per_review = {int(k): v for k, v in existing.items()}
        log.info("  Loaded %d existing results from %s", len(aspects_per_review), out_path)

    # ── filter to only unprocessed reviews ────────────────────
    processed_ids = set(aspects_per_review.keys())
    mask = ~df.index.isin(processed_ids)
    df_new = df[mask]
    log.info("  Skipping %d already-processed, running %d new reviews", len(df) - len(df_new), len(df_new))

    if len(df_new) > 0:
        nlp = spacy.load("en_core_web_sm")
        texts = df_new["clean_text"].tolist()
        review_ids = df_new.index.tolist()

        for doc, rid in zip(nlp.pipe(texts, batch_size=100), review_ids):
            chunks = []
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                # remove leading determiners (the, a, an, my, this, etc.)
                phrase = re.sub(r"^(the|a|an|my|its|this|that|these|those)\s+", "", phrase)
                phrase = phrase.strip()
                if phrase and len(phrase) >= config.ABSA_MIN_ASPECT_LEN:
                    chunks.append(phrase)
            # deduplicate while preserving order
            seen = set()
            unique = []
            for c in chunks:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            aspects_per_review[rid] = unique

    # ── save combined output ──────────────────────────────────
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in aspects_per_review.items()}, f, indent=2)
    log.info("  Task 1 done — %d reviews total, saved → %s", len(aspects_per_review), out_path)

    sample_ids = list(aspects_per_review.keys())[:3]
    for sid in sample_ids:
        log.info("  [sample] review %s → %s", sid, aspects_per_review[sid][:5])

    return aspects_per_review


# ──────────────────────────────────────────────────────────────
#  TASK 2 — Aspect-level Sentiment (zero-shot NLI)
# ──────────────────────────────────────────────────────────────
def extract_absa_triplets(df: pd.DataFrame, aspects_per_review: dict) -> dict:
    """
    For each review, takes the aspects from Task 1 and classifies
    sentiment towards each aspect using a zero-shot NLI model.

    The prompt template: "The {aspect} is {positive/negative/neutral}"
    lets the NLI model score how well each sentiment label fits.

    Parameters
    ----------
    df : pd.DataFrame — must have 'clean_text' column
    aspects_per_review : dict — review_id → list of aspect strings

    Returns
    -------
    dict : review_id → list of {aspect, sentiment, confidence}
    """
    log.info("═══ TASK 2: ABSA Triplet Extraction (zero-shot NLI) ═══")

    # ── load existing results to resume ───────────────────────
    out_path = OUTPUT_DIR / "task2_absa_triplets.json"
    triplets_per_review = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        triplets_per_review = {int(k): v for k, v in existing.items()}
        log.info("  Loaded %d existing results from %s", len(triplets_per_review), out_path)

    # ── filter to only unprocessed reviews ────────────────────
    processed_ids = set(triplets_per_review.keys())
    all_review_ids = df.index.tolist()
    new_ids = [rid for rid in all_review_ids if rid not in processed_ids]
    log.info("  Skipping %d already-processed, running %d new reviews", len(all_review_ids) - len(new_ids), len(new_ids))

    if len(new_ids) > 0:
        classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1,
        )

        candidate_labels = ["positive", "negative", "neutral"]
        total = len(new_ids)

        for i, rid in enumerate(new_ids):
            text = df.loc[rid, "clean_text"]
            aspects = aspects_per_review.get(rid, [])
            triplets = []

            for asp in aspects:
                hypothesis_text = f"{text}"
                try:
                    result = classifier(
                        hypothesis_text,
                        candidate_labels,
                        hypothesis_template=f"The {asp} is {{}}.",
                    )
                    best_label = result["labels"][0]
                    best_score = result["scores"][0]
                    triplets.append({
                        "aspect":     asp,
                        "sentiment":  best_label,
                        "confidence": round(float(best_score), 4),
                    })
                except Exception as e:
                    log.warning("  Review %s, aspect '%s' failed: %s", rid, asp, str(e)[:60])

            triplets_per_review[rid] = triplets

            # save periodically every 50 reviews to avoid losing progress
            if (i + 1) % 50 == 0:
                log.info("  Processed %d / %d new reviews", i + 1, total)
                with open(out_path, "w") as f:
                    json.dump({str(k): v for k, v in triplets_per_review.items()}, f, indent=2)

    # ── save combined output ──────────────────────────────────
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in triplets_per_review.items()}, f, indent=2)
    log.info("  Task 2 done — %d reviews total, saved → %s", len(triplets_per_review), out_path)

    sample_ids = list(triplets_per_review.keys())[:3]
    for sid in sample_ids:
        log.info("  [sample] review %s → %s", sid, triplets_per_review[sid][:3])

    return triplets_per_review


# ──────────────────────────────────────────────────────────────
#  TASK 3 — Clean & Filter
# ──────────────────────────────────────────────────────────────
def clean_and_filter(triplets_per_review: dict) -> dict:
    """
    Cleans raw ABSA output:
      - Removes generic/useless aspects (config.ABSA_GENERIC_ASPECTS)
      - Removes aspects shorter than ABSA_MIN_ASPECT_LEN
      - Filters by confidence threshold (config.ABSA_MIN_CONFIDENCE)
      - Deduplicates aspects within each review
      - Normalizes text (lowercase, trim whitespace)

    Parameters
    ----------
    triplets_per_review : dict — review_id → list of {aspect, sentiment, confidence}

    Returns
    -------
    dict : review_id → list of {aspect, sentiment}
    """
    log.info("═══ TASK 3: Clean & Filter ═══")

    cleaned = {}
    total_before = 0
    total_after = 0

    for rid, triplets in triplets_per_review.items():
        total_before += len(triplets)
        seen = set()
        filtered = []

        for t in triplets:
            asp = t["aspect"].lower().strip()
            sent = t["sentiment"]
            conf = t.get("confidence", 1.0)

            # skip generic aspects
            if asp in config.ABSA_GENERIC_ASPECTS:
                continue
            # skip too-short aspects
            if len(asp) < config.ABSA_MIN_ASPECT_LEN:
                continue
            # skip low confidence
            if conf < config.ABSA_MIN_CONFIDENCE:
                continue
            # skip duplicates
            if asp in seen:
                continue

            seen.add(asp)
            filtered.append({"aspect": asp, "sentiment": sent})

        cleaned[rid] = filtered
        total_after += len(filtered)

    # ── save final output ─────────────────────────────────────
    out_path = OUTPUT_DIR / "task3_cleaned_aspects.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in cleaned.items()}, f, indent=2)

    log.info(
        "  Task 3 done — aspects: %d → %d (removed %d noise entries)",
        total_before, total_after, total_before - total_after,
    )
    log.info("  Saved → %s", out_path)

    sample_ids = list(cleaned.keys())[:3]
    for sid in sample_ids:
        log.info("  [sample] review %s → %s", sid, cleaned[sid][:3])

    return cleaned


# ──────────────────────────────────────────────────────────────
#  RUN ALL — entry point for Stage 3
# ──────────────────────────────────────────────────────────────
def run_absa_pipeline(df: pd.DataFrame) -> dict:
    """
    Runs the full ABSA pipeline (Tasks 1–3) on a sample of reviews.

    Parameters
    ----------
    df : pd.DataFrame — preprocessed reviews with 'clean_text' and 'label'

    Returns
    -------
    dict : review_id → [(aspect, sentiment)] — cleaned results
    """
    log.info("─── Stage 3: ABSA Pipeline ───")

    df_sample = df.copy()
    df_sample = df_sample.reset_index(drop=True)
    log.info("  Running ABSA on %d reviews (with resume support)", len(df_sample))

    # Task 1: spaCy aspect extraction
    aspects_spacy = extract_aspects_spacy(df_sample)

    # Task 2: Aspect-level sentiment via zero-shot NLI
    triplets = extract_absa_triplets(df_sample, aspects_spacy)

    # Task 3: Clean & filter
    cleaned = clean_and_filter(triplets)

    # ── summary stats ─────────────────────────────────────────
    reviews_with_aspects = sum(1 for v in cleaned.values() if len(v) > 0)
    total_aspects = sum(len(v) for v in cleaned.values())
    log.info("\n  ═══ ABSA Summary ═══")
    log.info("  Reviews processed:      %d", len(cleaned))
    log.info("  Reviews with aspects:   %d (%.1f%%)",
             reviews_with_aspects, 100 * reviews_with_aspects / max(len(cleaned), 1))
    log.info("  Total aspect-sentiments: %d", total_aspects)
    log.info("  Avg aspects/review:     %.1f", total_aspects / max(len(cleaned), 1))

    return cleaned


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    from preprocess import load_and_preprocess

    log.info("Loading data for ABSA...")
    df_train, df_val, df_test = load_and_preprocess()

    # run on test set
    run_absa_pipeline(df_test)
