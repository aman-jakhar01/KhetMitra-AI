
from modules.rag.document_loader import load_knowledge_documents
from modules.rag.embeddings import (
    generate_document_embeddings,
    generate_query_embedding,
    get_embedding_info
)

# Load existing knowledge documents
result = load_knowledge_documents()

documents = result["documents"]

print("Documents loaded:", len(documents))

# Generate document embeddings
vectors = generate_document_embeddings(documents)

print("Number of vectors:", len(vectors))
print("Vector dimensions:", len(vectors[0]))

# Generate a farmer question embedding
question = "How can I identify yellow rust in wheat?"

query_vector = generate_query_embedding(question)

print("Question vector dimensions:", len(query_vector))

# Model information
print("\nEmbedding model information:")
print(get_embedding_info())
