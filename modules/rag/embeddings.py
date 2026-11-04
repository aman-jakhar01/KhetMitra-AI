
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


# ==========================================
# 1. EMBEDDING MODEL CONFIGURATION
# ==========================================

MODEL_NAME = EMBEDDING_MODEL

MODEL_DEVICE = "cpu"


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load and cache the HuggingFace embedding model.

    Model:
    sentence-transformers/all-MiniLM-L6-v2

    The model converts text into 384-dimensional vectors.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,

        model_kwargs={
            "device": MODEL_DEVICE
        },

        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )

    return embedding_model


# ==========================================
# 3. GENERATE DOCUMENT EMBEDDINGS
# ==========================================

def generate_document_embeddings(documents):
    """
    Generate embeddings for a list of LangChain documents.
    """

    if not documents:
        raise ValueError(
            "No documents provided for embedding."
        )

    embedding_model = get_embedding_model()

    texts = [
        document.page_content
        for document in documents
    ]

    vectors = embedding_model.embed_documents(texts)

    return vectors


# ==========================================
# 4. GENERATE QUESTION EMBEDDING
# ==========================================

def generate_query_embedding(question):
    """
    Convert the farmer's question into an embedding vector.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    embedding_model = get_embedding_model()

    vector = embedding_model.embed_query(
        question.strip()
    )

    return vector


# ==========================================
# 5. EMBEDDING MODEL INFORMATION
# ==========================================

def get_embedding_info():

    return {
        "model_name": MODEL_NAME,
        "device": MODEL_DEVICE,
        "embedding_dimensions": 384,
        "normalized": True
    }
