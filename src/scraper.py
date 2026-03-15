"""
ViewZen Documentation Scraper
Scrapes all pages from the GitBook user guides using Playwright.
Falls back to Selenium if Playwright is unavailable.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── GitBook sitemap URL ──────────────────────────────────────────────────────
BASE_URL = "https://viewzenlabsug.gitbook.io/userguides"
SITEMAP_URL = f"{BASE_URL}/sitemap-pages.xml"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_from_html(html: str, url: str) -> dict:
    """Parse the page HTML and return structured content."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script / style / nav elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try to get the main content area (GitBook uses <main> or role=main)
    main = soup.find("main") or soup.find(attrs={"role": "main"})
    if main is None:
        main = soup.body or soup

    # Extract title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Build a clean text version
    paragraphs = []
    for elem in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "pre", "code"]):
        text = elem.get_text(" ", strip=True)
        if text:
            prefix = ""
            tag = elem.name
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag[1])
                prefix = "#" * level + " "
            elif tag == "li":
                prefix = "- "
            paragraphs.append(prefix + text)

    content = "\n\n".join(paragraphs)

    # Derive a section path from the URL
    section = url.replace(BASE_URL, "").strip("/")
    if not section:
        section = "home"

    return {
        "url": url,
        "title": title,
        "section": section,
        "content": content,
    }


def _get_urls_from_sitemap_html(html: str) -> list[str]:
    """Extract <loc> URLs from a sitemap XML string."""
    soup = BeautifulSoup(html, "lxml")
    return [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip().startswith("http")]


# ──────────────────────────────────────────────────────────────────────────────
# Playwright scraper (preferred)
# ──────────────────────────────────────────────────────────────────────────────

def scrape_with_playwright() -> list[dict]:
    """Scrape all documentation pages using Playwright (headless Chromium)."""
    from playwright.sync_api import sync_playwright

    results: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # 1. Fetch sitemap to discover all page URLs
        logger.info("Fetching sitemap: %s", SITEMAP_URL)
        page.goto(SITEMAP_URL, wait_until="networkidle", timeout=30_000)
        sitemap_html = page.content()
        urls = _get_urls_from_sitemap_html(sitemap_html)
        logger.info("Discovered %d pages from sitemap", len(urls))

        if not urls:
            logger.warning("Sitemap returned 0 URLs – falling back to manual list.")
            urls = [BASE_URL]

        # 2. Visit each page and extract content
        for idx, url in enumerate(urls, 1):
            logger.info("[%d/%d] Scraping: %s", idx, len(urls), url)
            try:
                page.goto(url, wait_until="networkidle", timeout=30_000)
                # Wait a bit for JS-rendered content
                page.wait_for_timeout(2000)
                html = page.content()
                doc = _extract_text_from_html(html, url)
                if doc["content"].strip():
                    results.append(doc)
                else:
                    logger.warning("Empty content for %s – skipping", url)
            except Exception as exc:
                logger.error("Failed to scrape %s: %s", url, exc)

        browser.close()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Selenium scraper (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def scrape_with_selenium() -> list[dict]:
    """Fallback scraper using Selenium + Chrome."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=chrome_options)
    results: list[dict] = []

    try:
        # 1. Fetch sitemap
        logger.info("Fetching sitemap with Selenium: %s", SITEMAP_URL)
        driver.get(SITEMAP_URL)
        time.sleep(3)
        sitemap_html = driver.page_source
        urls = _get_urls_from_sitemap_html(sitemap_html)
        logger.info("Discovered %d pages from sitemap", len(urls))

        if not urls:
            urls = [BASE_URL]

        # 2. Visit each page
        for idx, url in enumerate(urls, 1):
            logger.info("[%d/%d] Scraping: %s", idx, len(urls), url)
            try:
                driver.get(url)
                # Wait for body to load
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(3)  # Let JS render
                html = driver.page_source
                doc = _extract_text_from_html(html, url)
                if doc["content"].strip():
                    results.append(doc)
                else:
                    logger.warning("Empty content for %s – skipping", url)
            except Exception as exc:
                logger.error("Failed to scrape %s: %s", url, exc)
    finally:
        driver.quit()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def scrape_documentation(method: str = "playwright") -> list[dict]:
    """
    Scrape all ViewZen documentation pages.

    Args:
        method: 'playwright' (default) or 'selenium'

    Returns:
        List of dicts with keys: url, title, section, content
    """
    if method == "playwright":
        try:
            docs = scrape_with_playwright()
        except ImportError:
            logger.warning("Playwright not installed – falling back to Selenium")
            docs = scrape_with_selenium()
    else:
        docs = scrape_with_selenium()

    # Save raw scraped data
    output_dir = Path(settings.raw_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scraped_docs.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d documents to %s", len(docs), output_path)
    return docs


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scrape_documentation()
