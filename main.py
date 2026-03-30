# ══════════════════════════════════════════════════════════════
#  main.py
#  Pipeline entry point — runs all stages in order and prints
#  a final comparison table of every model's performance.
#
#  Run:
#    python main.py
#
#  View MLflow results:
#    mlflow ui    →  http://localhost:5000
# ══════════════════════════════════════════════════════════════
from pathlib import Path
import json
import logging
import warnings
from pathlib import Path

import mlflow

import config
from preprocess          import load_and_preprocess
from stage1_classical    import run_logistic_regression, run_xgboost
from stage2_transformers import run_zero_shot, run_distilbert

Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
def print_summary(results: dict):
    """
    Prints a formatted comparison table of all model results
    and highlights the best-performing model by macro F1.

    Parameters
    ----------
    results : dict mapping model name → metrics dict
              (each metrics dict must have 'accuracy' and 'macro_f1')
    """
    print("\n" + "═" * 57)
    print(f"  {'Model':<32}  {'Acc':>7}  {'Macro F1':>9}")
    print("═" * 57)
    for name, m in results.items():
        print(f"  {name:<32}  {m['accuracy']:>7.4f}  {m['macro_f1']:>9.4f}")
    print("═" * 57)

    best_name, best_m = max(results.items(), key=lambda x: x[1]["macro_f1"])
    print(f"\n  🏆  Best: {best_name}  (macro_f1 = {best_m['macro_f1']:.4f})\n")


# ──────────────────────────────────────────────────────────────
def save_results(results: dict):
    """
    Serialises the results dictionary to a JSON file under the
    configured output directory so downstream stages (ABSA,
    recommendation) can reference the winning model.

    Parameters
    ----------
    results : dict mapping model name → metrics dict
    """
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(config.OUTPUT_DIR) / "pipeline_results.json"

    # strip non-serialisable values before writing
    serialisable = {
        name: {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
        for name, m in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    log.info("Results saved → %s", out_path)


# ──────────────────────────────────────────────────────────────
def main():
    """
    Orchestrates the full pipeline:
      1. Preprocessing  — load, clean, and split the dataset
      2. Stage 1-A      — TF-IDF + Logistic Regression
      3. Stage 1-B      — TF-IDF + XGBoost
      4. Stage 2-A      — Zero-shot RoBERTa
      5. Stage 2-B      — Fine-tuned distilBERT
      6. Summary        — comparison table + JSON results file
    """
    # ── MLflow experiment setup ───────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXP)

    # ── Step 1: Load and preprocess data ──────────────────────
    log.info("─── Preprocessing ───")
    df_train, df_val, df_test = load_and_preprocess()

    results = {}

    # ── Step 2: Stage 1 — Classical ML ───────────────────────
    log.info("─── Stage 1: Classical ML Baselines ───")
    results["TF-IDF + LogReg"]  = run_logistic_regression(df_train, df_val, df_test)
    results["TF-IDF + XGBoost"] = run_xgboost(df_train, df_val, df_test)

    # ── Step 3: Stage 2 — Transformers ───────────────────────
    log.info("─── Stage 2: Transformer Models ───")
    results["Zero-shot RoBERTa"]     = run_zero_shot(df_test)
    results["Fine-tuned distilBERT"] = run_distilbert(df_train, df_val, df_test)

    # ── Step 4: Report ────────────────────────────────────────
    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
