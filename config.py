
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ============================================================
# 1. BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# 2. APPLICATION SETTINGS
# ============================================================

APP_NAME = "KhetMitra AI"
APP_VERSION = "1.0.0"
APP_TAGLINE = "AI-powered wheat farming decision support system"
DEFAULT_REGION = "Haryana"


# ============================================================
# 3. GROQ / LLM SETTINGS
# ============================================================

# Read from Streamlit Cloud Secrets first, then local .env
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = None

GROQ_API_KEY = GROQ_API_KEY or os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.8-27b"
)


# ============================================================
# 4. KNOWLEDGE BASE SETTINGS
# ============================================================

KNOWLEDGE_BASE_DIR = (
    BASE_DIR / "knowledge_base" / "wheat"
)

RAG_INDEX_DIR = (
    BASE_DIR / "data" / "rag_index"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4


# ============================================================
# 5. MODEL DIRECTORY
# ============================================================

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 6. CROP DISEASE MODEL
# ============================================================

WHEAT_DISEASE_MODEL_PATH = (
    MODEL_DIR / "wheat_resnet18_best.pth"
)

DISEASE_MODEL_PATH = WHEAT_DISEASE_MODEL_PATH


# ============================================================
# 7. WHEAT YIELD MODEL
# ============================================================

WHEAT_YIELD_MODEL_PATH = (
    MODEL_DIR / "wheat_yield_xgboost.joblib"
)

YIELD_MODEL_PATH = WHEAT_YIELD_MODEL_PATH


# ============================================================
# 8. DATA DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


# ============================================================
# 9. DIRECTORY INITIALIZATION
# ============================================================

DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    RAG_INDEX_DIR,
    MODEL_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 10. ENVIRONMENT VALIDATION
# ============================================================

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not configured.")

if not KNOWLEDGE_BASE_DIR.exists():
    print("WARNING: Wheat knowledge base directory not found.")

if not WHEAT_YIELD_MODEL_PATH.exists():
    print("WARNING: Wheat yield model file not found.")

if not WHEAT_DISEASE_MODEL_PATH.exists():
    print("WARNING: Wheat disease model file not found.")
