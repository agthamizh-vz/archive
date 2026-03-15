"""
RAG Chatbot – Retrieval-Augmented Generation using OpenAI + Pinecone
"""

import logging
from textwrap import dedent

from openai import OpenAI

from src.config import settings
from src.vector_store import query_vector_store

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""\
    You are the official ViewZen AI Assistant. Your job is to answer user
    questions about ViewZen products (Accounts, Appverse, Analytics,
    Messaging & Marketing Platform) using ONLY the documentation context
    provided below.

    Rules:
    1. Answer ONLY based on the provided context. If the context does not
       contain enough information, say so clearly.
    2. Do NOT make up features, settings, or workflows that are not in the
       context.
    3. Be concise but thorough. Use bullet points or numbered lists when
       appropriate.
    4. When referencing a specific documentation page, include the URL from
       the source metadata.
    5. If the user's question is ambiguous, ask a clarifying question.
    6. Always maintain a helpful and professional tone.
""")


def _build_context_block(retrieved_docs: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc["metadata"]
        source_url = meta.get("url", "N/A")
        title = meta.get("title", "N/A")
        parts.append(
            f"--- Source {i} (title: {title}, url: {source_url}) ---\n"
            f"{doc['text']}\n"
        )
    return "\n".join(parts)


def ask(query: str, top_k: int | None = None, chat_history: list[dict] | None = None) -> dict:
    """
    Answer a user query using RAG.

    Args:
        query: the user's question.
        top_k: number of chunks to retrieve (default from settings).
        chat_history: optional list of {"role": ..., "content": ...} messages.

    Returns:
        {
            "answer": str,
            "sources": [{"url": str, "title": str, "section": str}],
            "query": str,
        }
    """
    # 1. Retrieve relevant context
    retrieved = query_vector_store(query, top_k=top_k)
    context_block = _build_context_block(retrieved)

    # 2. Build the messages list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    user_message = (
        f"Context from ViewZen documentation:\n\n{context_block}\n\n"
        f"User question: {query}"
    )
    messages.append({"role": "user", "content": user_message})

    # 3. Call the LLM
    client = OpenAI(api_key=settings.openai_api_key)

    logger.info("Sending query to %s …", settings.llm_model)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    # 4. Collect source references (deduplicate by URL)
    seen_urls: set[str] = set()
    sources = []
    for doc in retrieved:
        url = doc["metadata"].get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "title": doc["metadata"].get("title", ""),
                "section": doc["metadata"].get("section", ""),
            })

    return {
        "answer": answer,
        "sources": sources,
        "query": query,
    }


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    import sys

    question = " ".join(sys.argv[1:]) or "What are the standard roles in ViewZen?"
    result = ask(question)
    print(f"\nQ: {result['query']}")
    print(f"\nA: {result['answer']}")
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['title']}: {s['url']}")
