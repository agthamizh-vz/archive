import json

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from src.config import settings

pc = Pinecone(api_key=settings.pinecone_api_key)
model = SentenceTransformer(settings.embedding_model)


def get_index():
    """Create Pinecone index if missing, then return it."""
    if settings.pinecone_index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            settings.pinecone_index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(settings.pinecone_index_name)


def build_vector_store():
    """Load chunks, embed them, and upsert into Pinecone."""
    with open("data/processed/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    index = get_index()

    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        embeddings = model.encode([c["text"] for c in batch], normalize_embeddings=True).tolist()

        index.upsert(
            [
                {
                    "id": f"chunk_{i + j}",
                    "values": emb,
                    "metadata": {**c["metadata"], "text": c["text"][:3800]},
                }
                for j, (emb, c) in enumerate(zip(embeddings, batch))
            ]
        )


def query_vector_store(query, top_k=None):
    """Query Pinecone and return matched chunks with metadata and score."""
    index = get_index()
    embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
    results = index.query(vector=embedding, top_k=top_k or settings.top_k, include_metadata=True)

    return [
        {
            "text": m.metadata.pop("text", ""),
            "metadata": m.metadata,
            "score": m.score,
        }
        for m in results.matches
    ]