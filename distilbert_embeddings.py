import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = "startup_nlp_features.csv"

OUTPUT_EMBEDDINGS = "startup_distilbert_embeddings.npy"
OUTPUT_IDS = "startup_embedding_ids.csv"

MODEL_NAME = "distilbert-base-uncased"

# If your computer is slow / low on RAM, use 8 instead
BATCH_SIZE = 16

# Maximum number of words/tokens read from each startup text
MAX_LENGTH = 128


# ============================================================
# 2. SELECT CPU OR GPU
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print("\nGPU detected!")
    print("Using GPU.")

else:

    device = torch.device("cpu")

    print("\nNo NVIDIA GPU detected.")
    print("Using CPU.")
    print("This is completely okay, but it may take longer.")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading NLP dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded!")
print("Shape:", df.shape)


# ============================================================
# 4. CHECK TEXT COLUMN
# ============================================================

if "nlp_text" not in df.columns:

    raise ValueError(
        "nlp_text column not found. "
        "Run nlp_feature_extraction.py first."
    )


# Make sure there are no missing text values
df["nlp_text"] = (
    df["nlp_text"]
    .fillna("")
    .astype(str)
)


# ============================================================
# 5. LOAD DISTILBERT TOKENIZER
# ============================================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded!")


# ============================================================
# 6. LOAD DISTILBERT MODEL
# ============================================================

print("\nLoading DistilBERT model...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(device)

# We are generating embeddings, not training yet
model.eval()

print("DistilBERT model loaded!")


# ============================================================
# 7. FUNCTION TO CREATE EMBEDDINGS
# ============================================================

def generate_embeddings(texts):

    all_embeddings = []

    for start in tqdm(
        range(
            0,
            len(texts),
            BATCH_SIZE
        ),
        desc="Generating embeddings"
    ):

        # Select one batch
        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        # ----------------------------------------------------
        # TOKENIZATION
        # ----------------------------------------------------

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        # Move tokens to CPU/GPU
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        # ----------------------------------------------------
        # DISTILBERT
        # ----------------------------------------------------

        # No gradients because we are not training yet
        with torch.no_grad():

            outputs = model(**encoded)

        # Shape:
        # batch size × tokens × 768

        last_hidden_state = (
            outputs.last_hidden_state
        )

        # ----------------------------------------------------
        # MEAN POOLING
        # ----------------------------------------------------
        #
        # DistilBERT gives one vector for every token.
        # We need one vector for the whole startup.
        #
        # Mean pooling combines the token vectors into
        # ONE 768-dimensional startup vector.
        # ----------------------------------------------------

        attention_mask = encoded[
            "attention_mask"
        ]

        mask = attention_mask.unsqueeze(
            -1
        ).expand(
            last_hidden_state.size()
        ).float()

        summed = torch.sum(
            last_hidden_state * mask,
            dim=1
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        embeddings = summed / counts

        # Move result back to CPU
        embeddings = (
            embeddings
            .cpu()
            .numpy()
        )

        all_embeddings.append(
            embeddings
        )

    return np.vstack(
        all_embeddings
    )


# ============================================================
# 8. GENERATE EMBEDDINGS
# ============================================================

print("\nStarting DistilBERT embedding generation...")

texts = df[
    "nlp_text"
].tolist()

embeddings = generate_embeddings(
    texts
)


# ============================================================
# 9. CHECK RESULT
# ============================================================

print("\nEmbedding generation complete!")

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "Expected:",
    "(50000, 768)"
)


# ============================================================
# 10. SAVE EMBEDDINGS
# ============================================================

np.save(
    OUTPUT_EMBEDDINGS,
    embeddings
)

print(
    "\nEmbeddings saved:",
    OUTPUT_EMBEDDINGS
)


# ============================================================
# 11. SAVE STARTUP IDs
# ============================================================

id_df = df[
    [
        "startup_id",
        "startup_name"
    ]
].copy()

id_df.to_csv(
    OUTPUT_IDS,
    index=False
)

print(
    "Startup IDs saved:",
    OUTPUT_IDS
)


# ============================================================
# 12. DISPLAY FIRST EMBEDDING
# ============================================================

print("\nFirst startup:")

print(
    df[
        "startup_name"
    ].iloc[0]
)

print("\nFirst embedding:")

print(
    embeddings[0]
)

print(
    "\nNumber of values in one embedding:",
    len(embeddings[0])
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("DISTILBERT EMBEDDINGS COMPLETE")
print("==========================================")

print(
    "Startups processed:",
    len(df)
)

print(
    "Embedding dimensions:",
    embeddings.shape[1]
)

print(
    "\nCreated files:"
)

print(
    "1.",
    OUTPUT_EMBEDDINGS
)

print(
    "2.",
    OUTPUT_IDS
)

print("\nNext step:")
print("Semantic similarity search")
print("Then DistilBERT fine-tuning.")