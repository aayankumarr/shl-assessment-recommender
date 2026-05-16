"""
SHL Catalog Scraper
-------------------
One-time script. Run this to build catalog.json, which the agent uses at runtime.

Why Playwright instead of requests:
  The SHL catalog page loads its table data via JavaScript. A plain HTTP request
  (requests library) only gets the raw HTML shell — no table content. Playwright
  launches a real headless Chromium browser, executes the JavaScript, waits for
  the table to appear, then hands us the fully-rendered HTML. After that,
  BeautifulSoup parses it exactly as before.

Three-step process:
  Step 1  - Scrape 32 listing pages → name, url, test_type, remote_testing, adaptive_irt
  Step 2  - Visit each detail page  → description, job_levels, languages, duration_minutes
  Step 2b - If a Product Fact Sheet PDF link exists, download + extract its text too

Install:
  pip install playwright beautifulsoup4 pdfplumber
  playwright install chromium
"""

import json
import time
import re
import sys
import io

import requests
import pdfplumber
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_URL    = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/products/product-catalog/"

# Used only for PDF downloads (Playwright handles HTML pages)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# ── Browser helper ─────────────────────────────────────────────────────────────

def get_soup(page: Page, url: str, wait_for: str = "table") -> BeautifulSoup:
    """
    Navigate to a URL with Playwright and return a BeautifulSoup object.

    page      : the single reusable Playwright browser page
    url       : the page to load
    wait_for  : text or CSS selector to wait for before reading the HTML.

    Why networkidle:
      "domcontentloaded" fires when raw HTML is parsed — too early, because the
      catalog table is injected by JavaScript AFTER that event. "networkidle"
      waits until all network requests (including the AJAX call that fetches
      the table data) have finished. Only then do we read the HTML.
    """
    page.goto(url, wait_until="networkidle", timeout=60_000)
    # Belt-and-suspenders: also wait for the specific element we care about
    try:
        page.wait_for_selector(wait_for, timeout=20_000)
    except Exception:
        pass  # networkidle should be sufficient; this is just an extra safety net
    html = page.content()
    return BeautifulSoup(html, "html.parser")


# ── Helpers ────────────────────────────────────────────────────────────────────

def expand_test_types(code: str) -> list[str]:
    """Turn 'KP' into ['Knowledge & Skills', 'Personality & Behavior']."""
    return [TEST_TYPE_LABELS[c] for c in code if c in TEST_TYPE_LABELS]


# ── Step 1: Listing pages ──────────────────────────────────────────────────────

def scrape_listing_page(page: Page, start: int) -> list[dict]:
    """
    Scrape one paginated listing page of Individual Test Solutions.

    URL pattern : /products/product-catalog/?start=0&type=1
    type=1      : Individual Test Solutions  (what we want)
    type=2      : Pre-packaged Job Solutions (out of scope)

    We know type=1 vs type=2 because the pagination links rendered by the page
    itself use these values — discovered by inspecting the live page with Firecrawl.

    Each page shows 12 rows. 32 pages total → ~384 assessments.
    """
    url  = f"{CATALOG_URL}?start={start}&type=1"
    # Wait specifically for the text "Individual Test Solutions" to appear —
    # this is the table header injected by JavaScript. More reliable than
    # waiting for any generic <table> element.
    soup = get_soup(page, url, wait_for="text=Individual Test Solutions")

    assessments = []

    # Find the Individual Test Solutions table. It always contains a <th> with
    # that exact text. We check all tables and pick the right one.
    target_table = None
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if any("Individual Test Solutions" in h for h in headers):
            target_table = table
            break

    if target_table is None:
        # Debug: print a slice of the raw HTML so we can see what actually loaded
        raw_preview = soup.get_text()[:500].replace("\n", " ").strip()
        print(f"\n    WARNING: table not found. Page preview: {raw_preview[:200]}")
        return assessments

    for row in target_table.find_all("tr")[1:]:   # row[0] is the header
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        link = cols[0].find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "")
        full_url = href if href.startswith("http") else BASE_URL + href

        # Columns 1 & 2: SHL renders a green dot / checkmark image when supported
        remote_testing = bool(
            cols[1].find("img") or
            cols[1].find(attrs={"aria-label": True}) or
            cols[1].get_text(strip=True) not in ("", "-")
        )
        adaptive_irt = bool(
            cols[2].find("img") or
            cols[2].find(attrs={"aria-label": True}) or
            cols[2].get_text(strip=True) not in ("", "-")
        )

        test_type_code = cols[3].get_text(strip=True)

        assessments.append({
            "name":             name,
            "url":              full_url,
            "remote_testing":   remote_testing,
            "adaptive_irt":     adaptive_irt,
            "test_type":        test_type_code,
            "test_type_labels": expand_test_types(test_type_code),
        })

    return assessments


# ── Step 2: Detail pages ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_url: str) -> str:
    """
    Download a PDF from a URL and extract all its text using pdfplumber.

    We use requests here (not Playwright) because PDFs are static binary files —
    no JavaScript needed. io.BytesIO wraps the bytes in a file-like object so
    pdfplumber can read it without saving anything to disk.
    """
    response = requests.get(pdf_url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        pages_text = [p.extract_text() for p in pdf.pages if p.extract_text()]

    return "\n".join(pages_text).strip()


def scrape_detail_page(page: Page, url: str) -> dict:
    """
    Scrape one assessment detail page.
    Extracts: description, job_levels, languages, duration_minutes.
    Also finds the Product Fact Sheet PDF link and extracts its text if present.
    """
    soup   = get_soup(page, url, wait_for="h4")
    detail = {}

    # Detail pages use <h4> tags as section labels followed by content siblings.
    for h4 in soup.find_all("h4"):
        label   = h4.get_text(strip=True)
        sibling = h4.find_next_sibling()
        if not sibling:
            continue
        content = sibling.get_text(strip=True)

        if "Description" in label:
            detail["description"] = content

        elif "Job levels" in label:
            detail["job_levels"] = [j.strip() for j in content.split(",") if j.strip()]

        elif "Languages" in label:
            detail["languages"] = [l.strip() for l in content.split(",") if l.strip()]

        elif "Assessment length" in label:
            # Text looks like: "Approximate Completion Time in minutes = 30"
            block = h4.find_parent().get_text() if h4.find_parent() else content
            m = re.search(r"=\s*(\d+)", block)
            if m:
                detail["duration_minutes"] = int(m.group(1))

        elif "Downloads" in label:
            # Walk siblings of the Downloads h4 looking for a "fact sheet" PDF link.
            # Stop if we hit the next h4 (next section).
            node = sibling
            while node:
                if node.name == "h4":
                    break
                if hasattr(node, "find"):
                    link = node.find("a")
                    if link and "fact sheet" in link.get_text(strip=True).lower():
                        href = link.get("href", "")
                        if href.endswith(".pdf"):
                            detail["fact_sheet_url"] = href
                node = node.find_next_sibling()

    # Download and parse the PDF if we found a link
    if "fact_sheet_url" in detail:
        try:
            detail["fact_sheet_text"] = extract_pdf_text(detail["fact_sheet_url"])
        except Exception as e:
            detail["fact_sheet_text"] = None
            print(f"\n      PDF failed ({e})", end=" ")

    return detail


# ── Main orchestrator ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SHL Individual Test Solutions Scraper")
    print("=" * 60)

    all_assessments = []

    # Launch a single headless Chromium browser for the entire run.
    # Reusing one browser page avoids the overhead of launching a new browser
    # for each of the ~416 URLs we need to visit.
    #
    # Why stealth_sync:
    #   Headless Chromium sets navigator.webdriver = true, which websites
    #   check to detect bots and return a 403. stealth_sync patches ~20 browser
    #   properties (webdriver flag, plugins list, languages, etc.) to make the
    #   browser indistinguishable from a real user's Chrome.
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        browser_page = context.new_page()
        # Patch browser properties that websites check to detect headless bots.
        # This is what playwright-stealth does internally — we do it directly
        # to avoid library version issues.
        browser_page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'plugins',   { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        """)

        # ── Step 1: listing pages ──────────────────────────────────────────
        total_pages = 32
        print(f"\nSTEP 1: Scraping {total_pages} listing pages...")

        for page_num, start in enumerate(range(0, total_pages * 12, 12), start=1):
            print(f"  Page {page_num:2d}/{total_pages} (start={start:3d}) ...", end=" ", flush=True)
            try:
                items = scrape_listing_page(browser_page, start)
                all_assessments.extend(items)
                print(f"{len(items)} items")
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(0.5)   # polite delay between requests

        # Deduplicate by URL
        seen, unique = set(), []
        for a in all_assessments:
            if a["url"] not in seen:
                seen.add(a["url"])
                unique.append(a)
        all_assessments = unique

        print(f"\nTotal unique assessments: {len(all_assessments)}")

        if not all_assessments:
            print("\nERROR: No assessments found. Check your internet connection or the site structure.")
            browser.close()
            sys.exit(1)

        # ── Step 2: detail pages ───────────────────────────────────────────
        print(f"\nSTEP 2: Scraping detail pages (+ PDFs where available)...")

        for i, assessment in enumerate(all_assessments, start=1):
            label = assessment["name"][:50]
            print(f"  [{i:3d}/{len(all_assessments)}] {label:<50}", end=" ", flush=True)
            try:    
                detail = scrape_detail_page(browser_page, assessment["url"])
                assessment.update(detail)
                pdf_tag = " +PDF" if detail.get("fact_sheet_text") else ""
                print(f"ok{pdf_tag}")
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(0.5)

        browser.close()

    # ── Save ──────────────────────────────────────────────────────────────
    output_file = "catalog.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_assessments, f, indent=2, ensure_ascii=False)

    # Sanity check
    with_desc = sum(1 for a in all_assessments if a.get("description"))
    with_pdf  = sum(1 for a in all_assessments if a.get("fact_sheet_text"))

    print(f"\n{'=' * 60}")
    print(f"Saved {len(all_assessments)} assessments to {output_file}")
    print(f"  With description : {with_desc}/{len(all_assessments)}")
    print(f"  With PDF text    : {with_pdf}/{len(all_assessments)}")
    print(f"{'=' * 60}")
    print("\nSample entry:")
    if all_assessments:
        sample = all_assessments[0].copy()
        if sample.get("fact_sheet_text"):
            sample["fact_sheet_text"] = sample["fact_sheet_text"][:300] + "..."
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
