
from modules.rag.retriever import (
    retrieve_documents,
    format_retrieved_context,
    get_retrieval_info
)

print("\nKHETMITRA AI - RETRIEVER TEST")
print("=" * 45)

print("\nINDEX INFORMATION")
print(get_retrieval_info())

question = "What are the important practices for wheat cultivation?"

print("\nQUESTION:")
print(question)

print("\nRETRIEVED DOCUMENTS:")

documents = retrieve_documents(question, k=4)

for i, document in enumerate(documents, start=1):
    print(f"\n--- RESULT {i} ---")
    print("Source:", document["source"])
    print("Category:", document["category"])
    print("Page:", document["page"])
    print("Distance:", document["distance"])
    print("Content:", document["content"][:500])

print("\nFORMATTED CONTEXT")
print("=" * 45)
print(format_retrieved_context(documents))

print("\nRETRIEVER TEST COMPLETED")
