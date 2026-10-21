# ============================================================
# NLP FEATURE EXTRACTION FOR STARTUP DATA
# ============================================================
#
# This program:
# 1. Loads the cleaned startup dataset
# 2. Cleans startup text
# 3. Performs basic tokenization
# 4. Calculates basic NLP features
# 5. Extracts keywords using TF-IDF
# 6. Creates TF-IDF feature vectors
# 7. Saves the NLP-ready dataset
#
# ============================================================

import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


# ============================================================
# 1. FILE NAMES
# ============================================================

INPUT_FILE = "startup_cleaned_dataset.csv"

OUTPUT_DATA_FILE = "startup_nlp_features.csv"
OUTPUT_TFIDF_FILE = "startup_tfidf_features.npz"
OUTPUT_VOCAB_FILE = "tfidf_vocabulary.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 3. CHECK REQUIRED COLUMN
# ============================================================

if "startup_description" not in df.columns:

    raise ValueError(
        "startup_description column was not found."
    )


# ============================================================
# 4. NLP TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert everything to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove leading/trailing spaces
    text = text.strip()

    return text


print("\nCleaning startup descriptions...")

df["nlp_text"] = df["startup_description"].apply(
    clean_text
)


# ============================================================
# 5. TOKENIZATION
# ============================================================
#
# Tokenization means splitting text into individual words.
#
# Example:
#
# "AI startup in healthcare"
#
# becomes:
#
# ["ai", "startup", "in", "healthcare"]
#
# ============================================================

print("\nPerforming tokenization...")

df["tokens"] = df["nlp_text"].apply(
    lambda text: text.split()
)


# ============================================================
# 6. BASIC NLP FEATURES
# ============================================================

print("\nCreating basic NLP features...")


# Number of words
df["word_count"] = df["nlp_text"].apply(
    lambda text: len(text.split())
)


# Number of characters
df["character_count"] = df["nlp_text"].apply(
    len
)


# Average word length
df["average_word_length"] = (
    df["character_count"] /
    df["word_count"].replace(0, np.nan)
)

df["average_word_length"] = (
    df["average_word_length"].fillna(0)
)


# ============================================================
# 7. TF-IDF
# ============================================================
#
# TF-IDF = Term Frequency - Inverse Document Frequency
#
# It identifies words that are important for distinguishing
# one startup description from another.
#
# Example:
#
# "startup" appears in almost every description
# -> lower importance
#
# "blockchain" may appear in fewer descriptions
# -> higher importance
#
# ============================================================

print("\nCreating TF-IDF features...")


tfidf = TfidfVectorizer(

    # Convert text to lowercase
    lowercase=True,

    # Remove common English words
    stop_words="english",

    # Use individual words + two-word phrases
    ngram_range=(1, 2),

    # Keep maximum 5000 features
    max_features=5000,

    # Ignore words appearing in fewer than 2 documents
    min_df=2,

    # Ignore words appearing in more than 95% documents
    max_df=0.95
)


# Create TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(
    df["nlp_text"]
)


print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ============================================================
# 8. SAVE TF-IDF MATRIX
# ============================================================

save_npz(
    OUTPUT_TFIDF_FILE,
    tfidf_matrix
)

print(
    "TF-IDF matrix saved as:",
    OUTPUT_TFIDF_FILE
)


# ============================================================
# 9. SAVE TF-IDF VOCABULARY
# ============================================================

vocabulary = pd.DataFrame({

    "feature":
    tfidf.get_feature_names_out()

})


vocabulary.to_csv(
    OUTPUT_VOCAB_FILE,
    index=False
)


print(
    "TF-IDF vocabulary saved as:",
    OUTPUT_VOCAB_FILE
)


# ============================================================
# 10. EXTRACT TOP KEYWORDS
# ============================================================

print("\nExtracting top keywords...")


# Get all TF-IDF feature names
feature_names = np.array(
    tfidf.get_feature_names_out()
)


def get_top_keywords(row, top_n=10):

    # Convert sparse row to normal array
    scores = row.toarray().flatten()

    # Find features that actually have a score
    nonzero_indices = np.where(
        scores > 0
    )[0]

    # If there are no keywords
    if len(nonzero_indices) == 0:

        return ""


    # Sort according to TF-IDF score
    sorted_indices = nonzero_indices[
        np.argsort(
            scores[nonzero_indices]
        )
    ]


    # Take highest scoring words
    top_indices = sorted_indices[
        -top_n:
    ][::-1]


    # Convert indices to words
    keywords = feature_names[
        top_indices
    ]


    return ", ".join(keywords)


# Generate keywords for every startup

df["top_keywords"] = [

    get_top_keywords(
        tfidf_matrix[i],
        top_n=10
    )

    for i in range(
        tfidf_matrix.shape[0]
    )

]


# ============================================================
# 11. SAVE FINAL NLP DATASET
# ============================================================

df.to_csv(
    OUTPUT_DATA_FILE,
    index=False
)


print(
    "\nNLP dataset saved as:",
    OUTPUT_DATA_FILE
)


# ============================================================
# 12. DISPLAY SAMPLE RESULT
# ============================================================

print("\n")
print("==========================================")
print("NLP FEATURE EXTRACTION COMPLETE")
print("==========================================")


print("\nStartup:")
print(
    df["startup_name"].iloc[0]
)


print("\nOriginal description:")
print(
    df["startup_description"].iloc[0]
)


print("\nCleaned text:")
print(
    df["nlp_text"].iloc[0]
)


print("\nTokens:")
print(
    df["tokens"].iloc[0]
)


print("\nTop keywords:")
print(
    df["top_keywords"].iloc[0]
)


print("\nWord count:")
print(
    df["word_count"].iloc[0]
)


print("\nCharacter count:")
print(
    df["character_count"].iloc[0]
)


print("\nAverage word length:")
print(
    round(
        df["average_word_length"].iloc[0],
        2
    )
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("FILES CREATED")
print("==========================================")

print(
    "1. startup_nlp_features.csv"
)

print(
    "2. startup_tfidf_features.npz"
)

print(
    "3. tfidf_vocabulary.csv"
)


print("\nNext step:")
print(
    "DistilBERT embeddings + semantic similarity"
)

print("\nDone!")