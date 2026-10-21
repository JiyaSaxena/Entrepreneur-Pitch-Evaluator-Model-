# ============================================================
# STARTUP AI PROJECT - DATASET PREPARATION
# Jiya's Part: Data -> NLP -> Features -> Prediction Preparation
# ============================================================

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = "startup_valuation_dataset.csv"

OUTPUT_CLEAN_FILE = "startup_cleaned_dataset.csv"
OUTPUT_TRAIN_FILE = "startup_train.csv"
OUTPUT_VALID_FILE = "startup_validation.csv"
OUTPUT_TEST_FILE = "startup_test.csv"

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# ------------------------------------------------------------
# 3. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 4. REMOVE DUPLICATES
# ------------------------------------------------------------

before = len(df)
df = df.drop_duplicates()
after = len(df)

print("\nDuplicates removed:", before - after)

# ------------------------------------------------------------
# 5. HANDLE MISSING VALUES
# ------------------------------------------------------------

text_columns = [
    "startup_name",
    "country",
    "region",
    "industry",
    "funding_round",
    "lead_investor",
    "co_investors",
    "exit_type",
    "tags"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str)

numeric_columns = [
    "founded_year",
    "funding_amount_usd",
    "employee_count",
    "estimated_revenue_usd",
    "estimated_valuation_usd"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# ------------------------------------------------------------
# 6. CLEAN EXITED COLUMN
# ------------------------------------------------------------

if "exited" in df.columns:

    # Handles True/False, 1/0, Yes/No, etc.
    def convert_exited(value):
        value = str(value).strip().lower()

        if value in ["true", "1", "yes", "y"]:
            return 1

        if value in ["false", "0", "no", "n"]:
            return 0

        return 0

    df["exited"] = df["exited"].apply(convert_exited)

# ------------------------------------------------------------
# 7. CREATE INVESTMENT-WORTHINESS TARGET
# ------------------------------------------------------------
#
# We use successful exit as the observable outcome.
#
# 1 = Investment-worthy proxy
# 0 = Not investment-worthy proxy
#
# IMPORTANT:
# Do NOT use "exited" as an input feature later because it is
# exactly what creates the target.
# ------------------------------------------------------------

df["investment_worthy"] = df["exited"].astype(int)

print("\nInvestment-worthiness distribution:")
print(df["investment_worthy"].value_counts())

print("\nInvestment-worthiness percentage:")
print(
    df["investment_worthy"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 8. CREATE STARTUP DESCRIPTION
# ------------------------------------------------------------
#
# The original dataset does not contain a long description.
# Therefore, we create an NLP text field ONLY from information
# already present in the dataset.
#
# We are NOT inventing company facts.
# ------------------------------------------------------------

def create_description(row):

    startup = row.get("startup_name", "The startup")
    industry = row.get("industry", "Unknown")
    country = row.get("country", "Unknown")
    region = row.get("region", "Unknown")
    tags = row.get("tags", "Unknown")
    funding_round = row.get("funding_round", "Unknown")

    description = (
        f"{startup} is a startup operating in the {industry} industry. "
        f"It is based in {region}, {country}. "
        f"The startup is associated with the following focus areas: {tags}. "
        f"Its recorded funding stage is {funding_round}."
    )

    return description


df["startup_description"] = df.apply(create_description, axis=1)

# ------------------------------------------------------------
# 9. BASIC NLP TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep letters, numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_description"] = df["startup_description"].apply(clean_text)

# ------------------------------------------------------------
# 10. CREATE SIMPLE BUSINESS CATEGORY
# ------------------------------------------------------------
#
# This is a DERIVED feature, not a ground-truth label.
# It is inferred from industry/tags.
# ------------------------------------------------------------

def infer_business_category(row):

    text = (
        str(row.get("industry", "")) + " " +
        str(row.get("tags", ""))
    ).lower()

    if any(word in text for word in [
        "saas", "software", "cloud", "platform"
    ]):
        return "Software_Platform"

    elif any(word in text for word in [
        "e-commerce", "ecommerce", "marketplace", "retail"
    ]):
        return "Ecommerce_Marketplace"

    elif any(word in text for word in [
        "fintech", "finance", "bank", "payment", "insurance"
    ]):
        return "Financial_Technology"

    elif any(word in text for word in [
        "health", "medical", "biotech", "healthcare"
    ]):
        return "Healthcare"

    elif any(word in text for word in [
        "ai", "artificial intelligence", "machine learning", "ml"
    ]):
        return "AI_Technology"

    elif any(word in text for word in [
        "blockchain", "crypto", "web3"
    ]):
        return "Blockchain"

    elif any(word in text for word in [
        "logistics", "delivery", "supply chain"
    ]):
        return "Logistics"

    else:
        return "Other"


df["business_category"] = df.apply(
    infer_business_category,
    axis=1
)

# ------------------------------------------------------------
# 11. CREATE STARTUP AGE
# ------------------------------------------------------------

REFERENCE_YEAR = 2026

if "founded_year" in df.columns:

    df["startup_age"] = (
        REFERENCE_YEAR - df["founded_year"]
    )

    # Prevent impossible negative values
    df.loc[df["startup_age"] < 0, "startup_age"] = 0

# ------------------------------------------------------------
# 12. CREATE USEFUL FINANCIAL RATIOS
# ------------------------------------------------------------
#
# These are useful structured features for ML.
# Small epsilon avoids division by zero.
# ------------------------------------------------------------

epsilon = 1

if (
    "estimated_revenue_usd" in df.columns
    and "funding_amount_usd" in df.columns
):
    df["revenue_to_funding_ratio"] = (
        df["estimated_revenue_usd"] /
        (df["funding_amount_usd"] + epsilon)
    )

if (
    "estimated_valuation_usd" in df.columns
    and "funding_amount_usd" in df.columns
):
    df["valuation_to_funding_ratio"] = (
        df["estimated_valuation_usd"] /
        (df["funding_amount_usd"] + epsilon)
    )

if (
    "estimated_valuation_usd" in df.columns
    and "estimated_revenue_usd" in df.columns
):
    df["valuation_to_revenue_ratio"] = (
        df["estimated_valuation_usd"] /
        (df["estimated_revenue_usd"] + epsilon)
    )

# ------------------------------------------------------------
# 13. LOG-TRANSFORM LARGE FINANCIAL VALUES
# ------------------------------------------------------------
#
# Startup financial variables often have very large ranges.
# log1p makes them easier for ML/DL models to learn from.
# ------------------------------------------------------------

financial_columns = [
    "funding_amount_usd",
    "estimated_revenue_usd",
    "estimated_valuation_usd"
]

for col in financial_columns:

    if col in df.columns:

        df[f"log_{col}"] = np.log1p(
            df[col].clip(lower=0)
        )

# ------------------------------------------------------------
# 14. REMOVE POSSIBLE TARGET-LEAKAGE INPUTS
# ------------------------------------------------------------
#
# Do not delete them from the final dataset because they may be
# useful for reporting.
#
# But DO NOT feed these into the investment model:
#
# exited
# exit_type
# investment_worthy
#
# investment_worthy is the target.
# ------------------------------------------------------------

TARGET_COLUMN = "investment_worthy"

TEXT_COLUMN = "clean_description"

STRUCTURED_FEATURES = [
    "founded_year",
    "startup_age",
    "funding_amount_usd",
    "employee_count",
    "estimated_revenue_usd",
    "estimated_valuation_usd",
    "revenue_to_funding_ratio",
    "valuation_to_funding_ratio",
    "valuation_to_revenue_ratio",
    "log_funding_amount_usd",
    "log_estimated_revenue_usd",
    "log_estimated_valuation_usd",
    "country",
    "region",
    "industry",
    "funding_round",
    "business_category"
]

STRUCTURED_FEATURES = [
    col for col in STRUCTURED_FEATURES
    if col in df.columns
]

print("\nRecommended structured input features:")
print(STRUCTURED_FEATURES)

# ------------------------------------------------------------
# 15. TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------------------
#
# 70% training
# 15% validation
# 15% testing
#
# stratify keeps the same investment-worthy proportion in
# each split.
# ------------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df[TARGET_COLUMN]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df[TARGET_COLUMN]
)

print("\nDataset split:")
print("Training:", train_df.shape)
print("Validation:", validation_df.shape)
print("Testing:", test_df.shape)

# ------------------------------------------------------------
# 16. SAVE FILES
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_CLEAN_FILE,
    index=False
)

train_df.to_csv(
    OUTPUT_TRAIN_FILE,
    index=False
)

validation_df.to_csv(
    OUTPUT_VALID_FILE,
    index=False
)

test_df.to_csv(
    OUTPUT_TEST_FILE,
    index=False
)

print("\nFiles created successfully!")

print(OUTPUT_CLEAN_FILE)
print(OUTPUT_TRAIN_FILE)
print(OUTPUT_VALID_FILE)
print(OUTPUT_TEST_FILE)

# ------------------------------------------------------------
# 17. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========================================")
print("FINAL DATASET SUMMARY")
print("========================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nTarget:")
print(TARGET_COLUMN)

print("\nNLP text column:")
print(TEXT_COLUMN)

print("\nSample NLP description:")
print(df["startup_description"].iloc[0])

print("\nSample cleaned description:")
print(df["clean_description"].iloc[0])

print("\nPreparation completed.")
print("Next stages:")
print("1. NLP / keyword extraction")
print("2. DistilBERT embeddings")
print("3. Semantic similarity")
print("4. Domain classification")
print("5. Structured + BERT feature fusion")
print("6. Investment-worthiness prediction")
print("7. Evaluation: Precision, Recall, F1, ROC-AUC")
