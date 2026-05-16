"""
Retrieval evaluation: baseline (single query) vs query expansion (multi-query).

Usage:
  python evaluate.py --generate   # one-time: call LLM, save expansions.json
  python evaluate.py              # run evaluation using saved expansions.json
"""

import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()

# ── Ground truth ──────────────────────────────────────────────────────────────
# Each entry: (query, expected_assessment_names)
# Queries are hand-written to represent what rewrite_query would produce for
# the final confirmed recommendation in each sample conversation (C1–C10).
TEST_CASES = [
    (   # C1 — CXO/director selection, leadership benchmark
        "senior executive CXO director leadership personality selection benchmark",
        [
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ],
    ),
    (   # C2 — Senior Rust engineer, systems programming, networking
        "senior Rust engineer systems programming networking infrastructure live coding",
        [
            "Smart Interview Live Coding",
            "Linux Programming (General)",
            "Networking and Implementation (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (   # C3 — Entry-level contact centre agents, English US, high volume
        "entry level contact center agent English US customer service high volume screening",
        [
            "SVAR Spoken English (US) (New)",
            "Contact Center Call Simulation (New)",
            "Entry Level Customer Serv - Retail & Contact Center",
            "Customer Service Phone Simulation",
        ],
    ),
    (   # C4 — Graduate financial analysts, numerical reasoning, finance knowledge, SJT
        "graduate financial analyst numerical reasoning finance knowledge situational judgment",
        [
            "SHL Verify Interactive – Numerical Reasoning",
            "Financial Accounting (New)",
            "Basic Statistics (New)",
            "Graduate Scenarios",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (   # C5 — Sales organization reskilling, talent audit, development
        "sales organization reskilling talent audit development competencies personality",
        [
            "Global Skills Assessment",
            "Global Skills Development Report",
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ MQ Sales Report",
            "Sales Transformation 2.0 - Individual Contributor",
        ],
    ),
    (   # C6 — Plant operators chemical facility, safety critical, industrial
        "plant operator chemical facility safety critical dependability industrial personality",
        [
            "Manufac. & Indust. - Safety & Dependability 8.0",
            "Workplace Health and Safety (New)",
        ],
    ),
    (   # C7 — Healthcare admin bilingual, HIPAA, Spanish English hybrid battery
        "healthcare admin bilingual HIPAA patient records English Spanish personality dependability",
        [
            "HIPAA (Security)",
            "Medical Terminology (New)",
            "Microsoft Word 365 - Essentials (New)",
            "Dependability and Safety Instrument (DSI)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (   # C8 — Admin assistants Excel Word screening with simulation
        "admin assistant Microsoft Excel Word screening simulation knowledge skills",
        [
            "Microsoft Excel 365 (New)",
            "Microsoft Word 365 (New)",
            "MS Excel (New)",
            "MS Word (New)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (   # C9 — Senior backend Java Spring SQL AWS Docker engineer IC
        "senior backend Java Spring SQL AWS Docker engineer IC microservices personality",
        [
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (   # C10 — Graduate management trainee battery, cognitive + SJT (OPQ dropped)
        "graduate management trainee cognitive ability situational judgment battery",
        [
            "SHL Verify Interactive G+",
            "Graduate Scenarios",
        ],
    ),
]

EXPANSIONS_PATH = "expansions.json"

EXPAND_PROMPT = """You are a search query optimizer for SHL psychometric assessments.

Primary query: {query}

Generate 2-3 alternative phrasings of this query that approach the same need using different vocabulary.
Vary: job role terms, assessment type terms, seniority language.
Each phrasing must be under 20 words. Do NOT change the underlying intent.

Output a JSON array of strings only."""


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(results: list[dict], expected: list[str], k: int) -> float:
    retrieved = {r["name"] for r in results[:k]}
    hits = sum(1 for name in expected if name in retrieved)
    return hits / len(expected) if expected else 0.0


def mrr(results: list[dict], expected: list[str]) -> float:
    expected_set = set(expected)
    for i, r in enumerate(results, 1):
        if r["name"] in expected_set:
            return 1.0 / i
    return 0.0


def hit_at_k(results: list[dict], expected: list[str], k: int) -> int:
    retrieved = {r["name"] for r in results[:k]}
    return int(any(name in retrieved for name in expected))


# ── Generation ────────────────────────────────────────────────────────────────

def _llm_expand(query: str) -> list[str]:
    """Call Gemini; fall back to Groq on quota errors."""
    from google import genai
    from google.genai import types
    from groq import Groq

    prompt = EXPAND_PROMPT.format(query=query)

    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
            ),
        )
        result = json.loads(response.text)
        if isinstance(result, list):
            return [q for q in result if isinstance(q, str)][:3]
    except Exception as e:
        if not any(s in str(e) for s in ("429", "RESOURCE_EXHAUSTED", "QuotaFailure")):
            raise

    # Groq fallback
    groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Always respond with valid JSON only."},
            {"role": "user",   "content": prompt + '\n\nRespond with JSON only: {"queries": [...]}'},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    result = json.loads(completion.choices[0].message.content)
    queries = result.get("queries", [])
    return [q for q in queries if isinstance(q, str)][:3]


def generate_expansions():
    expansions = {}

    for i, (query, _) in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {query[:65]}...")
        try:
            expansions[query] = _llm_expand(query)
            print(f"  -> {expansions[query]}")
        except Exception as e:
            print(f"  ERROR: {e}")
            expansions[query] = []

    with open(EXPANSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(expansions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved expansions to {EXPANSIONS_PATH}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate():
    if not os.path.exists(EXPANSIONS_PATH):
        print(f"ERROR: {EXPANSIONS_PATH} not found. Run with --generate first.")
        return

    from retrieval import load_index, retrieve, retrieve_multi

    with open(EXPANSIONS_PATH, encoding="utf-8") as f:
        expansions = json.load(f)

    index, metadata, bm25, bi_encoder, cross_encoder = load_index()
    kwargs = dict(index=index, metadata=metadata, bm25=bm25,
                  bi_encoder=bi_encoder, cross_encoder=cross_encoder)

    rows = []
    for query, expected in TEST_CASES:
        baseline  = retrieve(query, top_k=10, **kwargs)
        all_queries = [query] + expansions.get(query, [])
        expanded  = retrieve_multi(all_queries, top_k=10, **kwargs)

        rows.append({
            "query":       query[:52],
            "r10_base":    recall_at_k(baseline, expected, 10),
            "r10_exp":     recall_at_k(expanded,  expected, 10),
            "mrr_base":    mrr(baseline, expected),
            "mrr_exp":     mrr(expanded,  expected),
            "hit5_base":   hit_at_k(baseline, expected, 5),
            "hit5_exp":    hit_at_k(expanded,  expected, 5),
        })

    _print_table(rows)


def _print_table(rows: list[dict]):
    header = f"{'Query':<54} {'R@10 B':>7} {'R@10 E':>7} {'MRR B':>7} {'MRR E':>7} {'H@5 B':>6} {'H@5 E':>6}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for r in rows:
        delta_r  = r["r10_exp"] - r["r10_base"]
        delta_m  = r["mrr_exp"] - r["mrr_base"]
        mark_r   = "^" if delta_r > 0 else ("v" if delta_r < 0 else " ")
        mark_m   = "^" if delta_m > 0 else ("v" if delta_m < 0 else " ")
        print(
            f"{r['query']:<54} "
            f"{r['r10_base']:>6.2f} "
            f"{r['r10_exp']:>6.2f}{mark_r} "
            f"{r['mrr_base']:>6.2f} "
            f"{r['mrr_exp']:>6.2f}{mark_m} "
            f"{r['hit5_base']:>5} "
            f"{r['hit5_exp']:>5}"
        )

    print(sep)
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in
           ["r10_base", "r10_exp", "mrr_base", "mrr_exp", "hit5_base", "hit5_exp"]}
    print(
        f"{'AVERAGE':<54} "
        f"{avg['r10_base']:>6.2f} "
        f"{avg['r10_exp']:>6.2f}  "
        f"{avg['mrr_base']:>6.2f} "
        f"{avg['mrr_exp']:>6.2f}  "
        f"{avg['hit5_base']:>5.2f} "
        f"{avg['hit5_exp']:>5.2f}"
    )
    print(sep)
    print("\nColumns: R@10 = Recall@10 | MRR = Mean Reciprocal Rank | H@5 = Hit@5")
    print("B = baseline (single query) | E = expanded (multi-query) | ^ improvement v regression\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate query expansion on SHL retrieval.")
    parser.add_argument("--generate", action="store_true",
                        help="Generate expansions.json via LLM (run once)")
    args = parser.parse_args()

    if args.generate:
        generate_expansions()
    else:
        evaluate()
