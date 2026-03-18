from src.scraper import scrape_documentation
from src.processor import process_documents
from src.vector_store import build_vector_store


def main():
    """Run full indexing pipeline in 3 steps."""
    print("Step 1/3: Scraping...")
    docs = scrape_documentation()
    print(f"{len(docs)} pages scraped")

    print("Step 2/3: Processing...")
    chunks = process_documents()
    print(f"{len(chunks)} chunks created")

    print("Step 3/3: Building vector store...")
    build_vector_store()
    print("Done!")


if __name__ == "__main__":
    main()