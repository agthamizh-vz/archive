from google import genai
from google.genai import types

from src.config import settings
from src.vector_store import query_vector_store

SYSTEM_PROMPT = """You are the ViewZen AI Assistant. Answer questions using ONLY
the documentation context provided. Be concise, use bullet points when helpful,
and cite source URLs. If context is insufficient, say so clearly."""


def ask(query, top_k=None):
    """Run one RAG cycle: retrieve docs -> generate answer -> return sources."""
    docs = query_vector_store(query, top_k=top_k)
    context = "\n\n".join(f"[{d['metadata']['title']}] {d['text']}" for d in docs)

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=settings.llm_model,
        contents=(
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        ),
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    seen = set()
    sources = []
    for d in docs:
        url = d["metadata"].get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append(
                {
                    "url": url,
                    "title": d["metadata"].get("title", ""),
                    "section": "",
                }
            )

    return {"query": query, "answer": (response.text or "").strip(), "sources": sources}