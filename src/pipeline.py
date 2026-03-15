"""
Full pipeline: Scrape → Process → Build Vector Store
Run with:  python -m src.pipeline
"""

import argparse
import logging
import sys

from src.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ViewZen RAG – Data Pipeline")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping (use existing raw data)",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip processing (use existing chunks)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild the vector store even if it already has data",
    )
    parser.add_argument(
        "--scrape-method",
        choices=["playwright", "selenium"],
        default="playwright",
        help="Scraping backend (default: playwright)",
    )
    args = parser.parse_args()

    # ── Step 1: Scrape ────────────────────────────────────────────────────
    if not args.skip_scrape:
        logger.info("═══ STEP 1/3: Scraping documentation ═══")
        from src.scraper import scrape_documentation

        docs = scrape_documentation(method=args.scrape_method)
        logger.info("Scraped %d documents", len(docs))
    else:
        logger.info("═══ STEP 1/3: Skipping scrape (--skip-scrape) ═══")

    # ── Step 2: Process ───────────────────────────────────────────────────
    if not args.skip_process:
        logger.info("═══ STEP 2/3: Processing & chunking ═══")
        from src.processor import process_documents

        chunks = process_documents()
        logger.info("Produced %d chunks", len(chunks))
    else:
        logger.info("═══ STEP 2/3: Skipping process (--skip-process) ═══")
        chunks = None

    # ── Step 3: Build vector store ────────────────────────────────────────
    logger.info("═══ STEP 3/3: Building Pinecone vector store ═══")
    from src.vector_store import build_vector_store

    build_vector_store(chunks=chunks, force_rebuild=args.force_rebuild)

    logger.info("✅ Pipeline complete!")


if __name__ == "__main__":
    main()
