r"""Master training and evaluation pipeline runner.

Executes the end-to-end workflow:
1. Loads raw datasets from D:/jiya/NLP&DL without modifying originals
2. Cleans datasets and extracts pre-decision features (preventing data leakage)
3. Generates canonical pitch dataset and formal data dictionary
4. Creates stratified train/validation/test splits (70/15/15, seed=42)
5. Fits Domain Classifier, Structured Model, Text Model, DistilBERT, and Fusion Model
6. Generates reference semantic embeddings with all-MiniLM-L6-v2
7. Evaluates all models, saves confusion matrices, ROC curves, error analysis, and comparison tables
8. Logs all experiment runs to experiments.json
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Ensure src is in python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.data.loader import load_raw_datasets
from src.data.cleaner import clean_shark_tank_india, clean_shark_tank_us, clean_startup_valuation
from src.data.canonical import create_canonical_dataset, generate_data_dictionary
from src.data.splitter import create_train_val_test_splits
from src.nlp.embeddings import EmbeddingGenerator
from src.models.domain_classifier import DomainClassifier
from src.models.structured_model import StructuredDealClassifier
from src.models.text_model import TextDealClassifier
from src.models.distilbert_classifier import DistilBertDealClassifier
from src.models.fusion_classifier import MultimodalFusionClassifier
from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)
from src.evaluation.comparison import ModelComparator
from src.evaluation.error_analysis import perform_error_analysis


def run_pipeline(config_path: str = "configs/config.yaml"):
    start_time = time.time()
    print("=" * 70)
    print("      STARTUP PITCH EVALUATOR - FULL TRAINING PIPELINE")
    print("=" * 70)

    # Load configuration
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = CURRENT_DIR / config_path
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["paths"]["raw_data_dir"])
    processed_dir = CURRENT_DIR / config["paths"]["processed_data_dir"]
    models_dir = CURRENT_DIR / config["paths"]["models_dir"]
    embeddings_dir = CURRENT_DIR / config["paths"]["embeddings_dir"]
    eval_dir = CURRENT_DIR / config["paths"]["evaluation_dir"]
    exp_dir = CURRENT_DIR / config["paths"]["experiments_dir"]

    for d in [processed_dir, models_dir, embeddings_dir, eval_dir, exp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD RAW DATASETS
    # ---------------------------------------------------------
    print("\n[Step 1/8] Loading raw datasets...")
    raw_data = load_raw_datasets(raw_dir)

    # ---------------------------------------------------------
    # 2. DATA CLEANING & LEAKAGE ISOLATION
    # ---------------------------------------------------------
    print("\n[Step 2/8] Cleaning datasets and isolating leakage...")
    sti_clean, sti_audit = clean_shark_tank_india(raw_data["shark_tank_india"])
    stp_clean, stp_audit = clean_shark_tank_us(raw_data["shark_tank_us"])
    # Sample 5000 rows from the 50k startup valuation dataset for fast, practical processing
    sv_sampled = raw_data["startup_valuation"].sample(n=min(5000, len(raw_data["startup_valuation"])), random_state=42)
    sv_clean, sv_audit = clean_startup_valuation(sv_sampled)

    # Save cleaned subsets
    sti_clean.to_csv(processed_dir / "shark_tank_india_clean.csv", index=False)
    stp_clean.to_csv(processed_dir / "shark_tank_us_clean.csv", index=False)
    sv_clean.to_csv(processed_dir / "startup_valuation_clean.csv", index=False)

    # ---------------------------------------------------------
    # 3. CREATE CANONICAL DATASET & DATA DICTIONARY
    # ---------------------------------------------------------
    print("\n[Step 3/8] Building canonical pitch dataset & data dictionary...")
    canonical_df = create_canonical_dataset(sti_clean, stp_clean, sv_clean)
    canonical_path = processed_dir / "canonical_pitches.csv"
    canonical_df.to_csv(canonical_path, index=False)
    print(f"  Canonical dataset saved: {canonical_df.shape[0]} rows to {canonical_path}")

    data_dict_df = generate_data_dictionary()
    dict_path = processed_dir / "data_dictionary.csv"
    data_dict_df.to_csv(dict_path, index=False)
    print(f"  Data dictionary saved: {len(data_dict_df)} features to {dict_path}")

    # ---------------------------------------------------------
    # 4. STRATIFIED SPLITTING (LEAKAGE PREVENTION)
    # ---------------------------------------------------------
    print("\n[Step 4/8] Creating stratified Train (70%), Val (15%), Test (15%) splits...")
    train_sti, val_sti, test_sti = create_train_val_test_splits(
        sti_clean,
        target_col="investment_decision",
        train_size=config["data"]["train_split"],
        val_size=config["data"]["val_split"],
        test_size=config["data"]["test_split"],
        random_state=config["project"]["random_seed"]
    )
    train_sti.to_csv(processed_dir / "train.csv", index=False)
    val_sti.to_csv(processed_dir / "val.csv", index=False)
    test_sti.to_csv(processed_dir / "test.csv", index=False)

    # Also split STUS for text models
    train_stp, val_stp, test_stp = create_train_val_test_splits(
        stp_clean,
        target_col="investment_decision",
        random_state=config["project"]["random_seed"]
    )

    # ---------------------------------------------------------
    # 5. GENERATE REFERENCE SEMANTIC EMBEDDINGS
    # ---------------------------------------------------------
    print("\n[Step 5/8] Generating sentence embeddings for reference startup search index...")
    embedder = EmbeddingGenerator(model_name=config["nlp"]["embedding_model"])
    ref_pitches = sti_clean["pitch_text"].tolist()
    ref_embeddings = embedder.generate_embeddings(ref_pitches, batch_size=32)
    embedder.save_embeddings(ref_embeddings, embeddings_dir / "startup_embeddings.npy")

    ref_metadata = sti_clean[["startup_name", "industry", "pitch_text", "investment_decision"]].copy()
    ref_metadata.to_csv(embeddings_dir / "startup_metadata.csv", index=False)

    # ---------------------------------------------------------
    # 6. MODEL TRAINING & FINE-TUNING
    # ---------------------------------------------------------
    print("\n[Step 6/8] Training Domain & Investment Classification Models...")
    comparator = ModelComparator()
    roc_dict = {}
    pr_dict = {}

    # --- 6A. Domain Classifier ---
    print("\n>>> Training Domain Classifier (TF-IDF + Logistic Regression)...")
    domain_clf = DomainClassifier(random_state=config["project"]["random_seed"])
    domain_clf.fit(train_sti["pitch_text"].tolist(), train_sti["industry"].tolist())
    domain_eval = domain_clf.evaluate(test_sti["pitch_text"].tolist(), test_sti["industry"].tolist())
    print(f"  Domain Classifier Test Accuracy: {domain_eval['accuracy']:.4f}, Macro F1: {domain_eval['macro_f1']:.4f}")
    domain_clf.save(models_dir / "domain_classifier.joblib")

    # --- 6B. Structured Financial Model ---
    print("\n>>> Training Structured Financial Baseline (Gradient Boosting)...")
    struct_clf = StructuredDealClassifier(model_type="gradient_boosting", random_state=config["project"]["random_seed"])
    struct_clf.fit(train_sti, target_col="investment_decision")
    struct_metrics = struct_clf.evaluate(test_sti, target_col="investment_decision")
    print(f"  Structured Model Test Metrics: {struct_metrics}")
    struct_clf.save(models_dir / "structured_model.joblib")
    comparator.add_model_result(
        "Structured Baseline (GBM)",
        "Pre-decision Financials (14 cols)",
        struct_metrics,
        "Pre-decision revenue, sales, margins, ask, equity, valuation"
    )
    y_test_sti = test_sti["investment_decision"].values
    probs_struct = struct_clf.predict_proba(test_sti)
    roc_dict["Structured Baseline"] = (y_test_sti, probs_struct)
    pr_dict["Structured Baseline"] = (y_test_sti, probs_struct)
    plot_confusion_matrix(
        y_test_sti,
        (probs_struct >= 0.5).astype(int),
        title="Structured Model Confusion Matrix",
        filepath=eval_dir / "confusion_matrix_structured.png"
    )

    # --- 6C. Text Deal Classifier (TF-IDF + Logistic Regression) ---
    print("\n>>> Training Text Deal Classifier (TF-IDF + Logistic Regression)...")
    text_clf = TextDealClassifier(
        min_df=config["nlp"]["min_df"],
        max_features=config["nlp"]["max_features"],
        random_state=config["project"]["random_seed"]
    )
    text_clf.fit(train_sti["pitch_text"].tolist(), train_sti["investment_decision"].tolist())
    text_metrics = text_clf.evaluate(test_sti["pitch_text"].tolist(), test_sti["investment_decision"].tolist())
    print(f"  Text Classifier Test Metrics: {text_metrics}")
    text_clf.save(models_dir / "text_model.joblib")
    comparator.add_model_result(
        "TF-IDF + Logistic Regression",
        "Pitch Text Unigrams + Bigrams",
        text_metrics,
        "Textual pitch unigrams & bigrams (1000 features)"
    )
    probs_text = text_clf.predict_proba(test_sti["pitch_text"].tolist())
    roc_dict["TF-IDF Text Model"] = (y_test_sti, probs_text)
    pr_dict["TF-IDF Text Model"] = (y_test_sti, probs_text)
    plot_confusion_matrix(
        y_test_sti,
        (probs_text >= 0.5).astype(int),
        title="TF-IDF Text Model Confusion Matrix",
        filepath=eval_dir / "confusion_matrix_text.png"
    )

    # --- 6D. Fine-tune DistilBERT ---
    print("\n>>> Fine-tuning DistilBERT on Pitch Text...")
    distilbert_clf = DistilBertDealClassifier(
        model_name=config["nlp"]["distilbert_model"],
        max_length=config["nlp"]["max_token_length"],
        batch_size=config["distilbert"]["batch_size"],
        learning_rate=config["distilbert"]["learning_rate"]
    )
    # Fine-tune for 2 epochs
    distilbert_clf.train(
        train_texts=train_sti["pitch_text"].tolist(),
        train_labels=train_sti["investment_decision"].tolist(),
        val_texts=val_sti["pitch_text"].tolist(),
        val_labels=val_sti["investment_decision"].tolist(),
        epochs=config["distilbert"]["epochs"],
        save_dir=models_dir / "distilbert"
    )
    distilbert_metrics = distilbert_clf.evaluate(
        test_sti["pitch_text"].tolist(),
        test_sti["investment_decision"].tolist()
    )
    print(f"  DistilBERT Test Metrics: {distilbert_metrics}")
    comparator.add_model_result(
        "Fine-tuned DistilBERT",
        "Transformer Sequence Classification",
        distilbert_metrics,
        "distilbert-base-uncased fine-tuned with early stopping"
    )
    probs_distilbert = distilbert_clf.predict_proba(test_sti["pitch_text"].tolist())
    roc_dict["DistilBERT"] = (y_test_sti, probs_distilbert)
    pr_dict["DistilBERT"] = (y_test_sti, probs_distilbert)
    plot_confusion_matrix(
        y_test_sti,
        (probs_distilbert >= 0.5).astype(int),
        title="DistilBERT Confusion Matrix",
        filepath=eval_dir / "confusion_matrix_distilbert.png"
    )

    # --- 6E. Multimodal Text + Structured Fusion Model ---
    print("\n>>> Training Multimodal Fusion Model (Embeddings + Structured Features)...")
    fusion_clf = MultimodalFusionClassifier(
        classifier_type="gradient_boosting",
        embedding_generator=embedder,
        random_state=config["project"]["random_seed"]
    )
    fusion_clf.fit(
        train_texts=train_sti["pitch_text"].tolist(),
        train_structured_df=train_sti,
        train_labels=train_sti["investment_decision"].tolist()
    )
    fusion_metrics = fusion_clf.evaluate(
        test_sti["pitch_text"].tolist(),
        test_sti,
        test_sti["investment_decision"].tolist()
    )
    print(f"  Multimodal Fusion Test Metrics: {fusion_metrics}")
    fusion_clf.save(models_dir / "fusion_model.joblib")
    comparator.add_model_result(
        "Text + Structured Fusion",
        "Dense Embedding (384) + Financials (14)",
        fusion_metrics,
        "Multimodal early fusion into Gradient Boosting Classifier"
    )
    probs_fusion = fusion_clf.predict_proba(test_sti["pitch_text"].tolist(), test_sti)
    roc_dict["Multimodal Fusion"] = (y_test_sti, probs_fusion)
    pr_dict["Multimodal Fusion"] = (y_test_sti, probs_fusion)
    plot_confusion_matrix(
        y_test_sti,
        (probs_fusion >= 0.5).astype(int),
        title="Multimodal Fusion Confusion Matrix",
        filepath=eval_dir / "confusion_matrix_fusion.png"
    )

    # ---------------------------------------------------------
    # 7. MODEL EVALUATION & ERROR ANALYSIS
    # ---------------------------------------------------------
    print("\n[Step 7/8] Generating comparative plots and conducting error analysis...")
    # Plot ROC & PR Curves
    plot_roc_curve(roc_dict, title="ROC Curves: Deal Prediction Models", filepath=eval_dir / "roc_curves.png")
    plot_precision_recall_curve(pr_dict, title="Precision-Recall Curves: Deal Prediction Models", filepath=eval_dir / "precision_recall_curves.png")

    # Save Comparison Table
    comparator.save_csv(eval_dir / "model_comparison.csv")
    print("\n" + comparator.to_markdown())

    # Systematic Error Analysis on Fusion Model
    errors_df, error_summary = perform_error_analysis(
        df=test_sti,
        y_true=y_test_sti,
        y_pred=(probs_fusion >= 0.5).astype(int),
        y_proba=probs_fusion,
        text_col="pitch_text",
        industry_col="industry",
        output_csv=eval_dir / "error_analysis_cases.csv"
    )
    with open(eval_dir / "error_analysis_summary.json", "w") as f:
        json.dump(error_summary, f, indent=2)

    print(f"\n[Error Analysis] Captured {len(errors_df)} errors on test set ({error_summary['error_rate']:.1%} error rate).")
    print(f"  - False Positives: {error_summary['false_positives']}")
    print(f"  - False Negatives: {error_summary['false_negatives']}")

    # ---------------------------------------------------------
    # 8. EXPERIMENT TRACKING
    # ---------------------------------------------------------
    print("\n[Step 8/8] Logging experiment records...")
    exp_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": config["project"]["random_seed"],
        "dataset_rows": len(sti_clean),
        "train_size": len(train_sti),
        "test_size": len(test_sti),
        "models_evaluated": comparator.records,
        "pipeline_duration_seconds": round(time.time() - start_time, 2)
    }
    with open(exp_dir / "experiments.json", "w") as f:
        json.dump([exp_log], f, indent=2)
    pd.DataFrame(comparator.records).to_csv(exp_dir / "experiments.csv", index=False)

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE IN {time.time() - start_time:.1f} SECONDS!")
    print(f"Artifacts saved in: {models_dir} and {eval_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
