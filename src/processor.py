import json
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def process_documents():
    """Load raw docs, chunk them, and save processed JSON."""
    with open("data/raw/scraped_docs.json", encoding="utf-8") as f:
        docs = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = []
    for doc in docs:
        text = re.sub(r"\n{3,}", "\n\n", doc["content"]).strip()

        for i, chunk in enumerate(splitter.split_text(text)):
            chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "url": doc["url"],
                        "title": doc["title"],
                        "chunk_index": i,
                    },
                }
            )

    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return chunks