
from modules.rag.qa_chain import answer_question, get_qa_info

print("\nKHETMITRA AI - RAG QUESTION ANSWERING TEST")
print("=" * 50)

print("\nMODEL INFORMATION")
print(get_qa_info())

question = "What are the important practices for wheat cultivation?"

print("\nQUESTION:")
print(question)

print("\nGENERATING ANSWER...\n")

result = answer_question(question)

print("ANSWER")
print("-" * 50)
print(result["answer"])

print("\nSOURCES")
print("-" * 50)

for source in result["sources"]:
    print(
        f"Document: {source['document']} | "
        f"Page: {source['page']} | "
        f"Category: {source['category']}"
    )

print("\nRAG TEST COMPLETED")
