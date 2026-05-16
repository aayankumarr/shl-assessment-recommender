"""
Catalog Preprocessor
--------------------
Reads catalog.json (raw scrape output) and writes catalog_processed.json.

What it does:
  1. Cleans fact_sheet_text  — strips copyright lines, page markers, www.shl.com noise
  2. Extracts structured fields from fact_sheet_text — job_roles, sector, scores, competencies
  3. Fills missing fields with safe defaults
  4. Builds a single search_text field per assessment — the field we embed for RAG
"""

import json
import re
import sys
from pathlib import Path

# Force UTF-8 output on Windows so special PDF characters don't crash the console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Cleaning ───────────────────────────────────────────────────────────────────

# Lines matching any of these patterns are pure noise from the PDF
NOISE_PATTERNS = [
    r"©.*SHL.*All rights reserved",   # copyright lines
    r"Page \d+ of \d+",               # page markers
    r"www\.shl\.com",                  # footer URLs
    r"Assessment Fact Sheet",          # repeated PDF header
    r"^\s*$",                          # blank lines
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


def clean_fact_sheet(text: str) -> str:
    """Remove boilerplate lines from raw PDF text."""
    if not text:
        return ""
    lines = text.splitlines()
    cleaned = [line for line in lines if not NOISE_RE.search(line)]
    return "\n".join(cleaned).strip()


# ── Extraction from fact_sheet_text ───────────────────────────────────────────

def extract_between(text: str, start_label: str, stop_labels: list[str]) -> str:
    """
    Extract the content between start_label and the first of stop_labels.
    Used to pull out specific sections from the semi-structured PDF text.
    """
    pattern = re.escape(start_label) + r"\s*(.*?)(?=" + "|".join(re.escape(s) for s in stop_labels) + r"|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def extract_job_roles(text: str) -> list[str]:
    """
    Pull 'Relevant Job Roles' or 'Job Family/Title' from fact sheet text.
    Returns a list of individual role strings.
    """
    # Try both label variants used across different fact sheets
    raw = (
        extract_between(text, "Relevant Job Roles", ["Details", "Language", "Average Testing"])
        or extract_between(text, "Job Family/Title", ["Details", "Language", "Average Testing"])
    )
    if not raw:
        return []
    # Roles are comma-separated but may wrap across lines — join first
    raw = " ".join(raw.split())
    return [r.strip() for r in raw.split(",") if r.strip()]


def extract_sector(text: str) -> str:
    """Pull the 'Sector' field from the fact sheet details table."""
    m = re.search(r"Sector\s+([^\n]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_scores(text: str) -> list[str]:
    """Pull 'Scores Reported' bullet points — the sub-scales the test measures."""
    raw = extract_between(text, "Scores Reported", ["O*NET", "Knowledge, Skills", "Competencies Measured", "©"])
    if not raw:
        return []
    # Each score is on its own line starting with "•"
    scores = [line.lstrip("•").strip() for line in raw.splitlines() if line.strip()]
    return [s for s in scores if s]


def extract_competencies(text: str) -> list[str]:
    """
    Pull the bullet list of topics the test covers.

    The consistent marker across all fact sheets is "The following areas are covered:"
    — this phrase always immediately precedes the bullet list regardless of whether
    the surrounding label says "Competencies Measured" or "Knowledge, Skills, Abilities".
    The older label variants are kept as fallbacks for PDFs that don't use this phrase.
    """
    raw = (
        extract_between(text, "The following areas are covered:", ["©", "Example", "Overview"])
        or extract_between(text, "Competencies\nMeasured", ["©", "Example"])
        or extract_between(text, "Competencies Measured",  ["©", "Example"])
    )
    if not raw:
        return []
    # Strip bullet characters — PDFs use both "•" (U+2022) and "" (Windows symbol font bullet)
    items = [re.sub(r"^[•\-\*]\s*", "", line).strip() for line in raw.splitlines() if line.strip()]
    LABEL_FRAGMENTS = {
        "abilities and", "competencies", "measured", "knowledge", "skills",
        "abilities", "and competencies", "knowledge skills",
        "the following areas are covered"
    }
    return [
        i for i in items
        if i and len(i) > 3 and i.lower() not in LABEL_FRAGMENTS
    ]


# ── search_text builder ────────────────────────────────────────────────────────

def build_search_text(a: dict, llm_desc: str = "") -> str:
    """
    Combine all useful fields into one rich text string for embedding.

    This is the single field the vector store embeds. It needs to be rich enough
    that a semantic query ("hiring a Java developer who works with stakeholders")
    finds the right assessments even when exact words don't match.

    Field ordering matters: put the most semantically important fields first
    since embedding models weight earlier tokens slightly more.
    """
    parts = []

    # Synthesis sentence in natural language at the top.
    # Embedding models are trained on prose so a sentence like
    # "Java 8 (New) is a Knowledge & Skills assessment for Java Developers"
    # matches conversational queries more strongly than key-value pairs alone.
    summary = [a["name"]]
    if a.get("test_type_labels"):
        summary.append("is a " + ", ".join(a["test_type_labels"]) + " assessment")
    if a.get("relevant_job_roles"):
        summary.append("for " + ", ".join(a["relevant_job_roles"][:3]))
    if a.get("sector"):
        summary.append("in the " + a["sector"] + " sector")
    if a.get("job_levels"):
        summary.append("suitable for " + ", ".join(a["job_levels"]))
    parts.append(" ".join(summary) + ".")

    # Name — exact identifier, always present
    parts.append(f"Assessment: {a['name']}")

    # Description — concise summary from the web page
    if a.get("description"):
        parts.append(f"Description: {a['description']}")

    # Relevant job roles — high signal for role-based queries
    if a.get("relevant_job_roles"):
        parts.append(f"Relevant job roles: {', '.join(a['relevant_job_roles'])}")

    # Test type — what kind of assessment this is
    if a.get("test_type_labels"):
        parts.append(f"Test type: {', '.join(a['test_type_labels'])}")

    # Job levels — seniority signal
    if a.get("job_levels"):
        parts.append(f"Job levels: {', '.join(a['job_levels'])}")

    # Sector — industry context
    if a.get("sector"):
        parts.append(f"Sector: {a['sector']}")

    # Competencies measured — the most detailed signal for what the test covers
    if a.get("competencies_measured"):
        parts.append(f"Competencies measured: {', '.join(a['competencies_measured'])}")

    # Scores reported — sub-scale names, useful for specific queries
    if a.get("scores_reported"):
        parts.append(f"Scores reported: {', '.join(a['scores_reported'])}")

    # Duration — occasionally queried ("short assessments only")
    if a.get("duration_minutes"):
        parts.append(f"Duration: {a['duration_minutes']} minutes")

    # Remote testing
    if a.get("remote_testing"):
        parts.append("Supports remote testing")

    # LLM-generated recruiter-language use-case description
    if llm_desc:
        parts.append(f"Use case: {llm_desc}")

    return "\n".join(parts)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    input_path  = Path("catalog.json")
    output_path = Path("catalog_processed.json")

    print(f"Reading {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        assessments = json.load(f)

    # Load LLM-generated use-case descriptions if available
    llm_descriptions = {}
    desc_path = Path("llm_descriptions.json")
    if desc_path.exists():
        with open(desc_path, encoding="utf-8") as f:
            llm_descriptions = json.load(f)
        print(f"Loaded LLM descriptions for {len(llm_descriptions)} assessments.")

    print(f"Processing {len(assessments)} assessments...")

    failed_extractions = []   # track entries where extraction yielded nothing

    for a in assessments:
        fact_text = a.get("fact_sheet_text") or ""

        # ── 1. Clean the raw PDF text ──
        a["fact_sheet_text_clean"] = clean_fact_sheet(fact_text)

        # ── 2. Extract structured fields from fact sheet ──
        a["relevant_job_roles"]    = extract_job_roles(fact_text)
        a["sector"]                = extract_sector(fact_text)
        a["scores_reported"]       = extract_scores(fact_text)
        a["competencies_measured"] = extract_competencies(fact_text)

        # ── 3. Detect silent extraction failures ──
        # If a fact sheet exists but ALL extractions returned empty, the labels
        # we expected weren't found — likely a new/different PDF format.
        # Log it so we can inspect the raw text and add new label variants.
        has_fact_sheet = bool(fact_text)
        all_empty = (
            not a["relevant_job_roles"] and
            not a["sector"] and
            not a["competencies_measured"]
        )
        if has_fact_sheet and all_empty:
            failed_extractions.append({
                "name":     a["name"],
                "url":      a["url"],
                "raw_text": fact_text[:300],   # first 300 chars to see what labels exist
            })

        # ── 4. Fill missing fields with safe defaults ──
        a.setdefault("description",      "")
        a.setdefault("job_levels",       [])
        a.setdefault("languages",        [])
        a.setdefault("duration_minutes", None)

        # ── 5. Build the unified search_text for embedding ──
        a["search_text"] = build_search_text(a, llm_desc=llm_descriptions.get(a["name"], ""))

    # Save processed catalog
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    # ── Stats ──
    with_roles  = sum(1 for a in assessments if a.get("relevant_job_roles"))
    with_sector = sum(1 for a in assessments if a.get("sector"))
    with_comp   = sum(1 for a in assessments if a.get("competencies_measured"))

    print(f"\nDone! Saved to {output_path}")
    print(f"  With relevant_job_roles    : {with_roles}/{len(assessments)}")
    print(f"  With sector                : {with_sector}/{len(assessments)}")
    print(f"  With competencies_measured : {with_comp}/{len(assessments)}")

    # ── Report silent failures ──
    if failed_extractions:
        print(f"\nWARNING: {len(failed_extractions)} entries had a fact sheet but extraction yielded nothing.")
        print("   These likely use different section labels. Raw text previews:")
        print("   (Add new label variants to the extraction functions for these)\n")
        for entry in failed_extractions:
            print(f"  Name : {entry['name']}")
            print(f"  URL  : {entry['url']}")
            print(f"  Text : {entry['raw_text']}")
            print()
    else:
        print("\n  All fact sheet extractions succeeded — no unknown label formats found.")

    print(f"\nSample search_text for '{assessments[1]['name']}':")
    print("-" * 60)
    print(assessments[1]["search_text"])
    print("-" * 60)


if __name__ == "__main__":
    main()
