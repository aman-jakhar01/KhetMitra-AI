
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import GROQ_API_KEY, GROQ_MODEL, RETRIEVAL_K

from modules.rag.retriever import (
    retrieve_documents,
    format_retrieved_context
)


SYSTEM_PROMPT = """
You are KhetMitra AI, an intelligent wheat farming
assistant designed primarily for farmers in Haryana, India.

Your task is to answer agricultural questions using the
provided retrieved documents.

STRICT KNOWLEDGE RULES:

1. Use the retrieved context as the primary and only
   factual source for your answer.

2. Never invent fertilizer quantities, pesticide doses,
   irrigation schedules, crop yields, government schemes,
   or agricultural statistics.

3. If the retrieved documents do not contain enough
   information, clearly say:
   "The available knowledge base does not contain
   enough information to answer this question reliably."

4. Never present national-level recommendations as
   Haryana-specific recommendations.

5. Do not confuse historical statistics with current
   agricultural conditions.

6. Do not treat an AI prediction as a confirmed diagnosis.

7. Explain agricultural concepts in simple language.

8. If a question involves chemical treatment, recommend
   checking current authorised product labels and seeking
   local agricultural expert advice.

ANSWER FORMAT:

Use this structure when appropriate:

### Answer
Give a direct answer to the farmer.

### Explanation
Explain the relevant agricultural information.

### Practical Recommendations
Provide only recommendations supported by the context.

### Sources
Mention the exact PDF filename and page number.

SOURCE CITATION RULES:

- Cite only documents present in the retrieved context.
- Preserve the original filename.
- Use the page number provided in the metadata.
- Never invent citations.
- If a claim is not supported, say so.

Your priority is agricultural accuracy and farmer safety.
"""


def get_llm():

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Check your project .env file."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1800,
        timeout=60,
        max_retries=2,
    )


def prepare_sources(documents):

    sources = []
    seen_sources = set()

    for document in documents:

        source = document.get(
            "source",
            "Unknown source"
        )

        page = document.get(
            "page",
            "Unknown page"
        )

        category = document.get(
            "category",
            "Unknown category"
        )

        source_key = (source, page)

        if source_key not in seen_sources:

            seen_sources.add(source_key)

            sources.append({
                "document": source,
                "page": page,
                "category": category
            })

    return sources


def answer_question(question: str):

    if not question or not question.strip():
        raise ValueError(
            "Please enter a valid farming question."
        )

    # 1. Retrieve relevant PDF passages
    documents = retrieve_documents(
        question=question.strip(),
        k=RETRIEVAL_K
    )

    if not documents:

        return {
            "answer": (
                "I could not find relevant information "
                "in the available farming documents."
            ),
            "sources": [],
            "retrieved_documents": [],
            "model": GROQ_MODEL
        }

    # 2. Prepare context
    context = format_retrieved_context(documents)

    # 3. Initialize Qwen through Groq
    llm = get_llm()

    # 4. Create model messages
    messages = [

        SystemMessage(
            content=SYSTEM_PROMPT
        ),

        HumanMessage(
            content=f"""
FARMER QUESTION:

{question}

RETRIEVED DOCUMENTS:

{context}

INSTRUCTIONS:

Answer the question using the retrieved documents.

If the documents do not contain sufficient evidence,
do not guess or add unsupported agricultural facts.

Cite supporting information using the exact PDF filename
and page number.

Make the answer clear and useful to an Indian farmer.
"""
        )
    ]

    # 5. Generate answer
    response = llm.invoke(messages)

    # 6. Extract sources
    sources = prepare_sources(documents)

    # 7. Return result
    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_documents": documents,
        "model": GROQ_MODEL
    }


def get_qa_info():

    return {
        "provider": "Groq",
        "model": GROQ_MODEL,
        "retrieval_method": "FAISS",
        "retrieval_k": RETRIEVAL_K,
        "temperature": 0.1
    }
