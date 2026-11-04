
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document

from config import BASE_DIR


# ==========================================
# KNOWLEDGE BASE CONFIGURATION
# ==========================================

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base" / "wheat"

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


# ==========================================
# 1. FIND KNOWLEDGE FILES
# ==========================================

def find_knowledge_files():

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE_DIR}"
        )

    files = [
        file
        for file in KNOWLEDGE_BASE_DIR.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


# ==========================================
# 2. EXTRACT PDF CONTENT
# ==========================================

def extract_pdf(file_path):

    documents = []

    reader = PdfReader(str(file_path))

    category = file_path.parent.name

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "category": category,
                    "page": page_number,
                    "file_path": str(file_path)
                }
            )
        )

    return documents


# ==========================================
# 3. EXTRACT TXT CONTENT
# ==========================================

def extract_txt(file_path):

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "category": file_path.parent.name,
                "page": 1,
                "file_path": str(file_path)
            }
        )
    ]


# ==========================================
# 4. LOAD ALL KNOWLEDGE DOCUMENTS
# ==========================================

def load_knowledge_documents():

    files = find_knowledge_files()

    if not files:
        raise ValueError(
            "No PDF or TXT documents found in the wheat knowledge base."
        )

    all_documents = []
    failed_files = []

    for file_path in files:

        try:

            if file_path.suffix.lower() == ".pdf":
                documents = extract_pdf(file_path)

            else:
                documents = extract_txt(file_path)

            all_documents.extend(documents)

        except Exception as error:

            failed_files.append({
                "file": str(file_path),
                "error": str(error)
            })

    if not all_documents:
        raise ValueError(
            "No readable text found in the knowledge documents."
        )

    return {
        "documents": all_documents,
        "total_files": len(files),
        "total_pages_or_sections": len(all_documents),
        "failed_files": failed_files
    }
