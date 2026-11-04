
from config import BASE_DIR
from modules.rag.document_loader import load_knowledge_documents
from modules.rag.vector_store import build_vector_store


def main():
    print("\n======================================")
    print("       KHETMITRA AI - RAG BUILDER")
    print("======================================\n")

    knowledge_dir = BASE_DIR / "knowledge_base" / "wheat"

    print(f"Knowledge directory: {knowledge_dir}")

    if not knowledge_dir.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {knowledge_dir}"
        )

    pdf_files = list(knowledge_dir.rglob("*.pdf"))

    print(f"\nPDF files discovered: {len(pdf_files)}")

    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    if not pdf_files:
        raise ValueError(
            "No PDF files found. Check your knowledge_base folders."
        )

    print("\n[1/3] Extracting PDF documents...")

    result = load_knowledge_documents()

    print(f"Files processed: {result['total_files']}")
    print(f"Pages/sections extracted: {result['total_pages_or_sections']}")

    if result["failed_files"]:
        print("\nFiles with extraction errors:")
        for item in result["failed_files"]:
            print(item["file"], ":", item["error"])

    print("\n[2/3] Building FAISS index...")
    
    index_result = build_vector_store(force_rebuild=True)

    print("\n[3/3] Index creation complete!")

    print("\nINDEX INFORMATION")
    print("------------------------------")
    print(index_result)

    print("\nYour RAG knowledge base is ready.")
    print("======================================\n")


if __name__ == "__main__":
    main()
