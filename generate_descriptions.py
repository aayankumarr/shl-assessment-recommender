"""
generate_descriptions.py
------------------------
One-time script: generates recruiter-language use-case descriptions for every
assessment in catalog_processed.json using Groq (llama-3.3-70b-versatile).

Saves results to llm_descriptions.json keyed by assessment name.
preprocess.py reads this file and appends the description to each search_text.

Resumes automatically if interrupted — already-generated entries are skipped.

Usage:
  python generate_descriptions.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from groq import Groq

CATALOG_PATH      = Path("catalog_processed.json")
DESCRIPTIONS_PATH = Path("llm_descriptions.json")
GROQ_MODEL        = "llama-3.3-70b-versatile"
SLEEP_BETWEEN     = 2.1  # seconds — keeps under 30 RPM free tier limit

PROMPT_TEMPLATE = """You are writing search index descriptions for an HR psychometric assessment catalog.

Assessment details:
- Name: {name}
- Type: {test_type}
- Job levels: {job_levels}
- Job roles: {job_roles}
- Sector: {sector}
- Competencies / areas covered: {competencies}
- Existing description: {description}

Write 2-3 sentences in recruiter language describing:
1. What hiring situation or role type this assessment is best suited for
2. What specific capability, trait, or knowledge area it measures
3. When a recruiter should choose this over alternatives

Rules:
- Be specific and concrete — avoid vague phrases like "suitable for many roles"
- Use recruiter vocabulary, not SHL jargon
- Do NOT repeat the assessment name
- Do NOT use bullet points or headers
- Output the sentences only — no preamble, no labels, no extra commentary"""


def _build_prompt(a: dict) -> str:
    return PROMPT_TEMPLATE.format(
        name=a["name"],
        test_type=", ".join(a.get("test_type_labels", [])) or "Not specified",
        job_levels=", ".join(a.get("job_levels", [])) or "Not specified",
        job_roles=", ".join(a.get("relevant_job_roles", [])[:5]) or "Not specified",
        sector=a.get("sector") or "Not specified",
        competencies=", ".join(a.get("competencies_measured", [])[:6]) or "Not specified",
        description=(a.get("description") or "")[:400],
    )


def main():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    # Resume from existing file if present — skips already-generated entries
    if DESCRIPTIONS_PATH.exists():
        with open(DESCRIPTIONS_PATH, encoding="utf-8") as f:
            descriptions = json.load(f)
        print(f"Resuming — {len(descriptions)}/{len(catalog)} already done.")
    else:
        descriptions = {}

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    todo  = [a for a in catalog if a["name"] not in descriptions]
    total = len(catalog)
    done  = total - len(todo)

    if not todo:
        print("All descriptions already generated.")
        return

    print(f"Generating descriptions for {len(todo)} assessments "
          f"(~{len(todo) * SLEEP_BETWEEN / 60:.1f} min at {SLEEP_BETWEEN}s/call)...\n")

    for i, a in enumerate(todo, 1):
        name = a["name"]
        print(f"[{done + i}/{total}] {name[:70]}")

        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You write concise, specific use-case descriptions for HR assessments. "
                                   "Output plain prose only — no bullet points, no labels.",
                    },
                    {"role": "user", "content": _build_prompt(a)},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            text = completion.choices[0].message.content.strip()
            descriptions[name] = text
            print(f"  {text[:90]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            descriptions[name] = ""

        # Save after every entry so progress survives interruption
        with open(DESCRIPTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

        time.sleep(SLEEP_BETWEEN)

    filled = sum(1 for v in descriptions.values() if v)
    print(f"\nDone. {filled}/{len(descriptions)} descriptions generated → {DESCRIPTIONS_PATH}")


if __name__ == "__main__":
    main()
