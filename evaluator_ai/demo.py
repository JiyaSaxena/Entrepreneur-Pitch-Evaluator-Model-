"""Demonstration script for Startup Pitch Evaluator AI.

Demonstrates the complete end-to-end evaluation pipeline on sample startup pitches:
- Domain Classification & Confidence
- Model-estimated Deal Likelihood
- Semantic Similarity against historical startups
- Multi-factor Transparent AI Score (0-100)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.nlp.embeddings import EmbeddingGenerator
from src.nlp.similarity import SimilarityEngine
from src.models.domain_classifier import DomainClassifier
from src.models.fusion_classifier import MultimodalFusionClassifier
from src.models.text_model import TextDealClassifier
from src.pipeline.pitch_evaluator import PitchEvaluatorPipeline, evaluate_startup


SAMPLE_PITCHES = [
    {
        "name": "MomoMelt FoodTech",
        "text": "MomoMelt creates ready-to-steam artisanal frozen momos and authentic Himalayan dipping sauces using 100% preservative-free, locally sourced ingredients for retail and quick-commerce delivery.",
        "industry": "Food and Beverage",
        "revenue": 180.0,
        "monthly_sales": 16.5,
        "gross_margin": 58.0,
        "net_margin": 18.0,
        "ebitda": 14.0,
        "cash_burn": 0,
        "has_patents": 0,
        "bootstrapped": 1,
        "skus": 8,
        "ask_amount": 60.0,
        "offered_equity": 5.0,
        "valuation_requested": 1200.0
    },
    {
        "name": "NeuroScan AI",
        "text": "NeuroScan develops an edge-AI portable medical imaging device that detects early-stage strokes and cranial trauma in rural clinics and emergency ambulances within 90 seconds.",
        "industry": "Medical/Health",
        "revenue": 45.0,
        "monthly_sales": 4.0,
        "gross_margin": 72.0,
        "net_margin": 10.0,
        "ebitda": 5.0,
        "cash_burn": 1,
        "has_patents": 1,
        "bootstrapped": 0,
        "skus": 2,
        "ask_amount": 100.0,
        "offered_equity": 4.0,
        "valuation_requested": 2500.0
    },
    {
        "name": "EcoVolt Mobility",
        "text": "EcoVolt builds smart solar-swappable battery stations and commercial electric cargo three-wheelers for last-mile hyper-local delivery fleets in tier-1 cities.",
        "industry": "Vehicles/Electrical Vehicles",
        "revenue": 220.0,
        "monthly_sales": 22.0,
        "gross_margin": 42.0,
        "net_margin": 8.0,
        "ebitda": 6.0,
        "cash_burn": 1,
        "has_patents": 1,
        "bootstrapped": 0,
        "skus": 3,
        "ask_amount": 120.0,
        "offered_equity": 6.0,
        "valuation_requested": 2000.0
    }
]


def load_pipeline() -> PitchEvaluatorPipeline:
    """Load pre-trained artifacts into evaluation pipeline."""
    models_dir = CURRENT_DIR / "artifacts" / "models"
    embeddings_dir = CURRENT_DIR / "artifacts" / "embeddings"

    embedder = EmbeddingGenerator()

    # Load Domain Classifier
    domain_clf_path = models_dir / "domain_classifier.joblib"
    domain_clf = DomainClassifier.load(domain_clf_path) if domain_clf_path.exists() else None

    # Load Fusion Classifier
    fusion_clf_path = models_dir / "fusion_model.joblib"
    fusion_clf = MultimodalFusionClassifier.load(fusion_clf_path, embedding_generator=embedder) if fusion_clf_path.exists() else None

    # Load Text Classifier
    text_clf_path = models_dir / "text_model.joblib"
    text_clf = TextDealClassifier.load(text_clf_path) if text_clf_path.exists() else None

    # Load Similarity Index
    embs_path = embeddings_dir / "startup_embeddings.npy"
    meta_path = embeddings_dir / "startup_metadata.csv"
    if embs_path.exists() and meta_path.exists():
        ref_embs = np.load(embs_path)
        ref_df = pd.read_csv(meta_path)
        sim_engine = SimilarityEngine(reference_df=ref_df, reference_embeddings=ref_embs, embedding_generator=embedder)
    else:
        sim_engine = None

    return PitchEvaluatorPipeline(
        domain_classifier=domain_clf,
        fusion_classifier=fusion_clf,
        text_classifier=text_clf,
        similarity_engine=sim_engine,
        embedding_generator=embedder
    )


def print_evaluation_report(startup_name: str, result: dict):
    """Format and print evaluation report according to specifications."""
    print("========================================")
    print("          PITCH EVALUATOR")
    print("========================================")
    print(f"Startup: {startup_name}")
    print("----------------------------------------")

    print("Predicted Domain:")
    print(result.get("predicted_domain", "Unknown"))
    print()

    print("Domain Confidence:")
    conf = result.get("domain_confidence", 0.0) * 100.0
    print(f"{conf:.1f}%")
    print()

    print("Investment Probability:")
    prob = result.get("investment_probability", 0.0) * 100.0
    print(f"{prob:.1f}%")
    print()

    print("Investment Prediction:")
    print(result.get("investment_prediction", "N/A"))
    print()

    print("Similar Startups:")
    sims = result.get("similar_startups", [])
    if sims:
        for idx, s in enumerate(sims[:3], start=1):
            s_name = s.get("startup_name", f"Startup {idx}")
            s_score = s.get("similarity_score", 0.0)
            s_outcome = s.get("investment_outcome", "")
            print(f"{idx}. {s_name} — {s_score:.2f} ({s_outcome})")
    else:
        print("None found or index not loaded.")
    print()

    print("Overall AI Score:")
    score = result.get("overall_score", 0.0)
    print(f"{int(round(score))}/100 ({result.get('rating_tier', '')})")
    print()

    print("Sub-Score Breakdown:")
    subs = result.get("subscores", {})
    for k, v in subs.items():
        label = k.replace("_score", "").replace("_", " ").title()
        print(f"  • {label}: {v}/100")

    print("========================================")
    print(result.get("disclaimer", ""))
    print("========================================\n")


def main():
    print("\nInitializing Pitch Evaluator System...")
    pipeline = load_pipeline()

    print(f"\nEvaluating {len(SAMPLE_PITCHES)} test startup pitches:\n")
    for sample in SAMPLE_PITCHES:
        res = pipeline.evaluate(
            pitch_text=sample["text"],
            industry=sample.get("industry"),
            revenue=sample.get("revenue", 0.0),
            monthly_sales=sample.get("monthly_sales", 0.0),
            gross_margin=sample.get("gross_margin", 50.0),
            net_margin=sample.get("net_margin", 15.0),
            ebitda=sample.get("ebitda", 10.0),
            cash_burn=sample.get("cash_burn", 0),
            has_patents=sample.get("has_patents", 0),
            bootstrapped=sample.get("bootstrapped", 1),
            skus=sample.get("skus", 5),
            ask_amount=sample.get("ask_amount", 50.0),
            offered_equity=sample.get("offered_equity", 5.0),
            valuation_requested=sample.get("valuation_requested", 1000.0)
        )
        print_evaluation_report(sample["name"], res)


if __name__ == "__main__":
    main()
