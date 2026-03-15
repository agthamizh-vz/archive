"""
Data Processing Pipeline
- Clean scraped text
- Chunk documents for embedding
- Store processed chunks
"""

import json
import logging
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove noise from scraped text."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove excessive whitespace within lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove GitBook UI artifacts
    artifacts = [
        r"Direct link to heading",
        r"CtrlK",
        r"\[Image:.*?\]",
        r"Copy\s*$",
        r"Last updated \d+ \w+ ago",
    ]
    for pattern in artifacts:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return text.strip()


def load_raw_docs(path: str | Path | None = None) -> list[dict]:
    """Load scraped documents from JSON."""
    if path is None:
        path = Path(settings.raw_data_dir) / "scraped_docs.json"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw docs not found at {path}. Run the scraper first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def process_documents(raw_docs: list[dict] | None = None) -> list[dict]:
    """
    Clean, chunk, and return processed document chunks.

    Each chunk dict contains:
        - text: chunk text
        - metadata: {url, title, section, chunk_index}
    """
    if raw_docs is None:
        raw_docs = load_raw_docs()

    logger.info("Processing %d raw documents", len(raw_docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []

    for doc in raw_docs:
        cleaned = clean_text(doc["content"])
        if not cleaned:
            continue

        # Prepend title and section as context header
        header = ""
        if doc.get("title"):
            header += f"# {doc['title']}\n"
        if doc.get("section"):
            section_parts = doc["section"].replace("-", " ").split("/")
            breadcrumb = " > ".join(part.title() for part in section_parts)
            header += f"Section: {breadcrumb}\n\n"

        full_text = header + cleaned

        text_chunks = splitter.split_text(full_text)

        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "url": doc["url"],
                    "title": doc.get("title", ""),
                    "section": doc.get("section", ""),
                    "chunk_index": i,
                },
            })

    logger.info("Created %d chunks from %d documents", len(chunks), len(raw_docs))

    # Persist processed chunks
    output_dir = Path(settings.processed_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info("Saved processed chunks to %s", output_path)
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    process_documents()
