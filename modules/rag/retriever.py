
from modules.rag.vector_store import load_vector_store
from config import RETRIEVAL_K


def retrieve_documents(question: str, k: int = RETRIEVAL_K):
    """
    Retrieve the most relevant passages from the FAISS knowledge base.

    Returns:
        A list of dictionaries containing:
        - content
        - source
        - category
        - page
        - distance
    """

    if not question or not question.strip():
        raise ValueError("Please enter a valid question.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        question.strip(),
        k=k
    )

    retrieved_documents = []

    for document, distance in results:
        metadata = document.metadata or {}

        retrieved_documents.append({
            "content": document.page_content,
            "source": metadata.get("source", "Unknown source"),
            "category": metadata.get("category", "Unknown category"),
            "page": metadata.get("page", "Unknown page"),
            "distance": float(distance)
        })

    return retrieved_documents


def format_retrieved_context(documents):
    """
    Format retrieved passages for the LLM.
    Preserve source names and page numbers for citations.
    """

    if not documents:
        return "No relevant information was found in the knowledge base."

    context_parts = []

    for index, document in enumerate(documents, start=1):
        context_parts.append(
            f"""
SOURCE {index}
Document: {document['source']}
Category: {document['category']}
Page: {document['page']}

Content:
{document['content']}
"""
        )

    return "\n\n".join(context_parts)


def get_retrieval_info():
    """
    Return information about the configured retriever.
    """

    vector_store = load_vector_store()

    return {
        "retrieval_method": "FAISS similarity search",
        "top_k": RETRIEVAL_K,
        "total_vectors": vector_store.index.ntotal
    }
