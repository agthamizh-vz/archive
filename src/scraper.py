import json
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE_URL = "https://viewzenlabsug.gitbook.io/userguides"


def scrape_documentation():
    """Scrape all docs listed in sitemap and save raw JSON."""
    results = []

    with sync_playwright() as pw:
        page = pw.chromium.launch().new_page()
        page.goto(f"{BASE_URL}/sitemap-pages.xml")

        urls = [loc.text for loc in BeautifulSoup(page.content(), "lxml").find_all("loc")]

        for url in urls:
            page.goto(url, wait_until="networkidle")
            soup = BeautifulSoup(page.content(), "lxml")

            title = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
            content = "\n\n".join(
                e.get_text(" ", strip=True)
                for e in soup.find_all(["h1", "h2", "p", "li"])
            )

            results.append({"url": url, "title": title, "content": content})

    with open("data/raw/scraped_docs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results