import json
import os
from typing import Literal, Type, Union

from google import genai
from google.genai import types
from groq import Groq
from pydantic import BaseModel

from retrieval import load_index, retrieve


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class Recommendation(BaseModel):
    name: str
    url: str = ""
    test_type: str = ""

class GenerateResponseOutput(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

class AgentState(BaseModel):
    history: list[Message]
    intent: str = ""
    rewritten_query: str = ""
    candidates: list[dict] = []
    reply: str = ""
    recommendations: list[dict] = []
    end_of_conversation: bool = False
    turn_count: int = 0


gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
groq_client   = Groq(api_key=os.environ["GROQ_API_KEY"])
GEMINI_MODEL  = "gemini-2.0-flash-lite"
GROQ_MODEL    = "llama-3.3-70b-versatile"

# Load retrieval index once at module import time (not per request)
_index, _metadata, _bm25, _bi_encoder, _cross_encoder = load_index()

# Name → URL lookup built from metadata — used to validate and fill URLs after LLM generation
# The LLM outputs assessment names; we look up the real URL so it can never hallucinate one
_name_to_url = {item["name"]: item["url"] for item in _metadata}

# Full catalog keyed by name — used to enrich candidates with job roles, competencies, etc.
with open("catalog_processed.json", encoding="utf-8") as _f:
    _catalog = {a["name"]: a for a in json.load(_f)}


def _enrich_candidate(c: dict) -> str:
    """Format a retrieved candidate with rich metadata for LLM context."""
    entry = _catalog.get(c["name"], {})
    lines = [f"- {c['name']}"]
    if entry.get("test_type_labels"):
        lines.append(f"  Type        : {', '.join(entry['test_type_labels'])}")
    if entry.get("job_levels"):
        lines.append(f"  Job levels  : {', '.join(entry['job_levels'])}")
    if entry.get("relevant_job_roles"):
        lines.append(f"  Job roles   : {', '.join(entry['relevant_job_roles'][:5])}")
    if entry.get("competencies_measured"):
        lines.append(f"  Competencies: {', '.join(entry['competencies_measured'][:6])}")
    if entry.get("sector"):
        lines.append(f"  Sector      : {entry['sector']}")
    if entry.get("duration_minutes"):
        lines.append(f"  Duration    : {entry['duration_minutes']} minutes")
    return "\n".join(lines)


def _call_llm(prompt: str, gemini_schema: Union[dict, Type[BaseModel]], groq_key: str = None):
    """
    Try Gemini with schema enforcement.
    On rate limit (429 / RESOURCE_EXHAUSTED), fall back to Groq with JSON mode.

    groq_key: for simple string responses, tells Groq to wrap its answer in
              {"<groq_key>": "..."} so we can extract it cleanly.
              Leave None for full object responses (generate_response).
    """
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=gemini_schema
            )
        )
        return json.loads(response.text)

    except Exception as e:
        if not any(s in str(e) for s in ("429", "RESOURCE_EXHAUSTED", "QuotaFailure")):
            raise

        # Groq fallback — JSON mode requires an object at the top level
        groq_prompt = prompt
        if groq_key:
            groq_prompt += f'\n\nRespond with JSON only: {{"{groq_key}": <your answer>}}'

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Always respond with valid JSON only."},
                {"role": "user",   "content": groq_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(completion.choices[0].message.content)
        return result[groq_key] if groq_key else result


def classify_intent(state: AgentState) -> AgentState:
    history_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in state.history
    )

    prompt = f"""You are analyzing a conversation about SHL assessments.
Based on the full conversation history, classify the user's latest intent.

Conversation:
{history_text}

Classify the intent as one of: recommend, refine, compare, clarify, refuse

Definitions:
- "clarify" → essential information is missing to make a useful recommendation. Choose this when:
    - No specific role or job function is mentioned (e.g. "senior leadership" alone is too vague)
    - Use case is unknown (selection vs development vs reskilling)
    - Seniority level is unclear and it would change the recommendation
    - Language or region is unspecified and it would affect the assessment choice
    - The request is too broad to retrieve meaningful results
  Be conservative — if in doubt between clarify and recommend, choose clarify.

- "recommend" → enough context exists to make a meaningful recommendation (role, purpose, or
  enough detail to retrieve relevant assessments). Only choose this when you have sufficient
  information.

- "refine" → user is explicitly modifying a previously recommended ASSESSMENT shortlist by
  naming specific assessments to add, remove, or swap.
  e.g. "Add Docker", "Drop REST", "Replace Verify G+ with something shorter"
  A user answering a question the agent asked is NEVER refine — it is clarify or recommend.
  "Selected" or "choosing" in the context of hiring/HR is NOT refine.

- "compare" → user is asking about differences between specific named assessments

- "refuse" → completely off-topic, harmful, or unrelated to assessment selection
  (e.g. legal advice, medical advice, general HR policy questions)"""

    intent = _call_llm(
        prompt,
        gemini_schema={"type": "STRING", "enum": ["recommend", "refine", "compare", "clarify", "refuse"]},
        groq_key="intent"
    )

    if intent not in {"recommend", "refine", "compare", "clarify", "refuse"}:
        intent = "clarify"
    state.intent = intent
    return state


def rewrite_query(state: AgentState) -> AgentState:
    history_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in state.history
    )

    prompt = f"""You are a search query optimizer for SHL assessments.

Conversation:
{history_text}

Rewrite the user's request as a single concise search query (max 30 words).
The query will be used to search a catalog of psychometric assessments.
Include: job role, skills being tested, seniority level, sector — whatever is mentioned.
Do NOT include filler words. Output the query string only."""

    state.rewritten_query = _call_llm(
        prompt,
        gemini_schema={"type": "STRING"},
        groq_key="query"
    )
    return state


def run_retrieval(state: AgentState) -> AgentState:
    candidates = retrieve(
        query=state.rewritten_query,
        top_k=10,
        index=_index,
        metadata=_metadata,
        bm25=_bm25,
        bi_encoder=_bi_encoder,
        cross_encoder=_cross_encoder,
    )
    state.candidates = candidates
    return state


def reflect_and_retry(state: AgentState) -> AgentState:
    candidates_text = "\n".join(_enrich_candidate(c) for c in state.candidates)

    history_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in state.history
    )

    prompt = f"""You are evaluating whether retrieved assessments match the user's need.

User conversation:
{history_text}

Retrieved assessments:
{candidates_text}

Do these results adequately address what the user is looking for?
Reply "yes" if they are relevant, "no" if they are clearly off-target."""

    verdict = _call_llm(
        prompt,
        gemini_schema={"type": "STRING", "enum": ["yes", "no"]},
        groq_key="verdict"
    )

    if verdict not in {"yes", "no"}:
        verdict = "yes"

    if verdict == "no":
        prompt2 = f"""The previous search query failed to find relevant SHL assessments.

Conversation:
{history_text}

Previous query: {state.rewritten_query}

Write a broader alternative search query (max 30 words) to try again."""

        state.rewritten_query = _call_llm(
            prompt2,
            gemini_schema={"type": "STRING"},
            groq_key="query"
        )
        state = run_retrieval(state)

    return state


def _finalize(state: AgentState, output: dict) -> AgentState:
    """Parse LLM output, validate URLs from catalog, drop hallucinated names."""
    parsed = GenerateResponseOutput.model_validate(output)
    validated = []
    for rec in parsed.recommendations:
        url = _name_to_url.get(rec.name)
        if url:
            validated.append({"name": rec.name, "url": url, "test_type": rec.test_type})
    state.reply = parsed.reply
    state.recommendations = validated
    state.end_of_conversation = parsed.end_of_conversation
    return state


def generate_clarification(state: AgentState) -> AgentState:
    """
    Intent: clarify
    Ask ONE targeted question. If enough context exists to make a partial
    recommendation, make it AND ask the remaining question in the same reply.
    """
    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in state.history)

    prompt = f"""You are an expert SHL assessment consultant. The user's request needs clarification
before you can retrieve and recommend assessments.

Conversation history:
{history_text}

Your job: ask exactly ONE targeted question — the single most important missing detail.
Common missing details: role or job function, use case (selection vs development vs reskilling),
seniority level, language or region.

NEVER ask multiple questions at once.

If you already have enough context to make a partial recommendation (role + seniority known),
give that partial recommendation AND ask the one remaining question in the same reply.

EXAMPLE — vague query, one question:
USER: "We need a solution for senior leadership."
→ {{"reply": "Happy to help narrow that down. Who is this meant for — external candidates being selected, or existing leaders receiving development feedback?", "recommendations": [], "end_of_conversation": false}}

EXAMPLE — partial recommend + follow-up:
USER: "CXOs and directors, 15+ years experience." [after prior clarify turn]
→ {{"reply": "For CXO/director-level, OPQ32r is the right instrument — it measures 32 workplace behaviour dimensions including strategic thinking and leadership. One question before I finalise the shortlist: is this for selection (comparing candidates) or development feedback for executives already in role?", "recommendations": [{{"name": "Occupational Personality Questionnaire OPQ32r", "url": "", "test_type": "P"}}], "end_of_conversation": false}}

Output names EXACTLY as in the SHL catalog. Leave url as empty string."""

    output = _call_llm(prompt, gemini_schema=GenerateResponseOutput)
    return _finalize(state, output)


def generate_recommendation(state: AgentState) -> AgentState:
    """
    Intent: recommend
    Produce a shortlist from retrieved candidates. Always include OPQ32r by default
    for selection scenarios. Acknowledge catalog gaps explicitly.
    """
    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in state.history)
    candidates_text = "\n".join(_enrich_candidate(c) for c in state.candidates) if state.candidates else "None"

    prompt = f"""You are an expert SHL assessment consultant recommending assessments for a recruiter.
URLs are filled automatically — output only exact assessment names from the candidates list.

Conversation history:
{history_text}

Retrieved assessment candidates:
{candidates_text}

RULES:
1. OPQ32r DEFAULT: For every hiring/selection scenario, include OPQ32r (Personality & Behavior)
   by default unless the user has explicitly excluded it. Say: "I'm including OPQ32r by default
   as the personality component — say the word if you'd rather drop it."

2. CATALOG GAP: If no candidate matches what the user asked for (e.g. a technology with no
   dedicated test), say explicitly: "SHL's catalog doesn't include a [X]-specific test."
   Then recommend the closest available alternatives from the candidates list.

3. TWO-STAGE BATTERY: For high-volume or multi-stage hiring, proactively frame recommendations
   as two stages — broad screen first, depth tests for shortlisted candidates.

4. end_of_conversation: true ONLY on explicit confirmation words: "perfect", "confirmed",
   "that works", "that's it", "locking it in", "that covers it", "done", "that's good".
   Never set true just because your answer sounds complete.

5. Only use names from the candidates list above. Never fabricate assessment names.

EXAMPLE — catalog gap:
USER: "I need a Rust programming assessment."
→ {{"reply": "SHL's catalog doesn't include a Rust-specific test. The closest alternatives for systems-level programming are Smart Interview Live Coding (where your panel can frame Rust tasks directly), Linux Programming, and Networking and Implementation.", "recommendations": [{{"name": "Smart Interview Live Coding", "url": "", "test_type": "K"}}, {{"name": "Linux Programming (General)", "url": "", "test_type": "K"}}, {{"name": "Networking and Implementation (New)", "url": "", "test_type": "K"}}], "end_of_conversation": false}}

EXAMPLE — end_of_conversation:
USER: "Perfect, that's what we need." → end_of_conversation: true, carry forward final shortlist
USER: "And can you add Docker?" → end_of_conversation: false  ← this is a modification, not confirmation"""

    output = _call_llm(prompt, gemini_schema=GenerateResponseOutput)
    return _finalize(state, output)


def generate_refinement(state: AgentState) -> AgentState:
    """
    Intent: refine
    Read the previous shortlist from conversation history. Apply the user's
    modification exactly — add/remove/swap. Never rebuild from scratch.
    """
    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in state.history)
    candidates_text = "\n".join(_enrich_candidate(c) for c in state.candidates) if state.candidates else "None"

    prompt = f"""You are an expert SHL assessment consultant updating an existing shortlist.
URLs are filled automatically — output only exact assessment names.

Conversation history:
{history_text}

Available candidates for new additions:
{candidates_text}

RULES:
1. Read the previous shortlist from the conversation history.
2. Apply the user's modification EXACTLY — add named items, remove named items, preserve everything else.
3. NEVER rebuild the list from scratch. All unmentioned items stay.
4. Return the COMPLETE updated list.
5. end_of_conversation: true ONLY on explicit confirmation: "perfect", "confirmed", "that works",
   "that's it", "locking it in", "that covers it", "done", "that's good".

EXAMPLE — exact add/drop:
Previous shortlist: [Core Java Advanced, Spring, SQL, RESTful Web Services, Verify G+, OPQ32r]
USER: "Add AWS and Docker. Drop REST."
→ {{"reply": "Updated — REST removed, AWS and Docker added:", "recommendations": [{{"name": "Core Java (Advanced Level) (New)", "url": "", "test_type": "K"}}, {{"name": "Spring (New)", "url": "", "test_type": "K"}}, {{"name": "SQL (New)", "url": "", "test_type": "K"}}, {{"name": "Amazon Web Services (AWS) Development (New)", "url": "", "test_type": "K"}}, {{"name": "Docker (New)", "url": "", "test_type": "K"}}, {{"name": "SHL Verify Interactive G+", "url": "", "test_type": "A"}}, {{"name": "Occupational Personality Questionnaire OPQ32r", "url": "", "test_type": "P"}}], "end_of_conversation": false}}
✗ WRONG: Ignoring history and building a new list from scratch."""

    output = _call_llm(prompt, gemini_schema=GenerateResponseOutput)
    return _finalize(state, output)


def generate_comparison(state: AgentState) -> AgentState:
    """
    Intent: compare
    Explain differences between named assessments. Carry forward the shortlist
    if the user is deciding between items in it; return empty if purely educational.
    """
    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in state.history)

    prompt = f"""You are an expert SHL assessment consultant explaining differences between assessments.

Conversation history:
{history_text}

RULES:
1. Explain the differences clearly and concisely — what each measures, scope, use case, norms.
2. If the user is DECIDING between items already in the current shortlist (choosing one over the
   other), carry the shortlist forward unchanged in recommendations.
3. If this is a general educational question (not a decision between shortlisted items),
   return an empty recommendations list.
4. end_of_conversation: true ONLY on explicit confirmation words.
5. Never fabricate assessment details. Only use what you know about SHL's catalog.

EXAMPLE — educational comparison, empty recommendations:
USER: "What's the difference between DSI and Safety & Dependability 8.0?"
→ {{"reply": "Both measure safety-relevant personality but at different levels. DSI is a standalone instrument used across sectors measuring integrity, reliability, and safety attitudes. The Safety & Dependability 8.0 is sector-specific with industrial norms — better for manufacturing contexts. If your facility is industrial-classified, the 8.0 gives you industry norms; otherwise DSI is the general-purpose choice.", "recommendations": [], "end_of_conversation": false}}

EXAMPLE — user deciding, carry shortlist forward:
Current shortlist has [OPQ32r, Verify G+, Graduate Scenarios]
USER: "Should we use OPQ32r or the MQ for personality?"
→ Explain difference, return [{{"name": "Occupational Personality Questionnaire OPQ32r", ...}}, {{"name": "SHL Verify Interactive G+", ...}}, ...] unchanged"""

    output = _call_llm(prompt, gemini_schema=GenerateResponseOutput)
    return _finalize(state, output)


def generate_refusal(state: AgentState) -> AgentState:
    """
    Intent: refuse
    Decline off-topic questions politely. Always keep end_of_conversation: false.
    """
    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in state.history)

    prompt = f"""You are an expert SHL assessment consultant. The user has asked something outside
your scope — legal advice, compliance interpretation, medical advice, or unrelated HR policy.

Conversation history:
{history_text}

RULES:
1. Politely decline the specific question.
2. Clarify what you CAN confirm about the assessment itself (features, what it measures).
3. Invite the user to continue with assessment selection questions.
4. ALWAYS set end_of_conversation: false — the user may have follow-up assessment questions.
5. Return empty recommendations.

EXAMPLE:
USER: "Are we legally required under HIPAA to test all staff who touch patient records?"
→ {{"reply": "That's a legal compliance question outside what I can advise on — your legal or compliance team is the right resource. What I can confirm is that HIPAA (Security) measures knowledge of HIPAA security provisions. Whether using it fulfils a particular regulatory obligation is a question for counsel. Is there anything else about the assessment selection I can help with?", "recommendations": [], "end_of_conversation": false}}"""

    output = _call_llm(prompt, gemini_schema=GenerateResponseOutput)
    state = _finalize(state, output)
    state.end_of_conversation = False  # hard override — refuse never ends conversation
    return state


def run_agent(history: list[dict]) -> dict:
    state = AgentState(
        history=[Message(**m) for m in history],
        turn_count=len([m for m in history if m["role"] == "user"])
    )

    # Hard stop at 8 turns
    if state.turn_count >= 8:
        return {
            "reply": "We've reached the maximum conversation length. Please start a new session.",
            "recommendations": [],
            "end_of_conversation": True
        }

    # Stage 1: classify intent
    state = classify_intent(state)

    # Stage 2: intent routing
    if state.intent == "clarify":
        state = generate_clarification(state)
    elif state.intent == "compare":
        state = generate_comparison(state)
    elif state.intent == "refuse":
        state = generate_refusal(state)
    else:
        # recommend / refine — need retrieval first
        state = rewrite_query(state)
        state = run_retrieval(state)
        state = reflect_and_retry(state)
        if state.intent == "recommend":
            state = generate_recommendation(state)
        else:
            state = generate_refinement(state)

    return {
        "reply": state.reply,
        "recommendations": state.recommendations,
        "end_of_conversation": state.end_of_conversation
    }
