"""
Vector Store – Pinecone with Sentence-Transformers embeddings
"""

import json
import logging
import time
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

# Singleton instances
_pinecone_index = None
_embedding_model: SentenceTransformer | None = None

INDEX_NAME = settings.pinecone_index_name
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-d vectors


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_pinecone_index():
    """Lazy-load (and create if needed) the Pinecone index."""
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    pc = Pinecone(api_key=settings.pinecone_api_key)

    # Create the index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        logger.info("Creating Pinecone index '%s' …", INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # Wait until the index is ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            logger.info("Waiting for index to be ready …")
            time.sleep(2)

    _pinecone_index = pc.Index(INDEX_NAME)
    logger.info("Connected to Pinecone index '%s'", INDEX_NAME)
    return _pinecone_index


def get_index_stats() -> dict:
    """Return current index statistics."""
    index = get_pinecone_index()
    stats = index.describe_index_stats()
    return {
        "total_vector_count": stats.total_vector_count,
        "dimension": stats.dimension,
    }


def build_vector_store(
    chunks: list[dict] | None = None,
    force_rebuild: bool = False,
) -> None:
    """
    Embed all document chunks and upsert into Pinecone.

    Args:
        chunks: list of {text, metadata} dicts. If None, loads from processed dir.
        force_rebuild: delete all vectors first.
    """
    index = get_pinecone_index()

    # Check if already populated
    stats = index.describe_index_stats()
    if not force_rebuild and stats.total_vector_count > 0:
        logger.info(
            "Index '%s' already has %d vectors – skipping rebuild. "
            "Pass force_rebuild=True to overwrite.",
            INDEX_NAME,
            stats.total_vector_count,
        )
        return

    if force_rebuild and stats.total_vector_count > 0:
        logger.info("Deleting all vectors from '%s' …", INDEX_NAME)
        index.delete(delete_all=True)
        time.sleep(2)

    # Load chunks if not provided
    if chunks is None:
        chunks_path = Path(settings.processed_data_dir) / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Processed chunks not found at {chunks_path}. Run the processor first."
            )
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)

    logger.info("Embedding %d chunks …", len(chunks))

    model = get_embedding_model()
    texts = [c["text"] for c in chunks]

    # Batch embed + upsert
    BATCH_SIZE = 100
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_chunks = chunks[i : i + BATCH_SIZE]

        embeddings = model.encode(
            batch_texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        vectors = []
        for j, (emb, chunk) in enumerate(zip(embeddings, batch_chunks)):
            vec_id = f"chunk_{i + j}"
            metadata = {
                **chunk["metadata"],
                "text": chunk["text"][:3800],  # Pinecone metadata limit ~40KB
            }
            vectors.append({"id": vec_id, "values": emb, "metadata": metadata})

        index.upsert(vectors=vectors)
        logger.info(
            "Upserted batch %d–%d / %d",
            i, min(i + BATCH_SIZE, len(texts)), len(texts),
        )

    # Small delay for Pinecone to index
    time.sleep(3)
    stats = index.describe_index_stats()
    logger.info("Vector store built with %d vectors", stats.total_vector_count)


def query_vector_store(query: str, top_k: int | None = None) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Returns list of dicts: {text, metadata, score}
    """
    if top_k is None:
        top_k = settings.top_k

    model = get_embedding_model()
    query_embedding = model.encode(
        [query], normalize_embeddings=True
    ).tolist()[0]

    index = get_pinecone_index()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    retrieved = []
    for match in results.matches:
        meta = dict(match.metadata)
        text = meta.pop("text", "")
        retrieved.append({
            "text": text,
            "metadata": meta,
            "score": match.score,
        })

    return retrieved


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    build_vector_store()
