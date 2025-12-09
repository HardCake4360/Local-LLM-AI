from typing import List, Dict, Any, Optional
import json, re

try:
    from llmClient import query_ollama
except Exception:
    query_ollama = None

DEFAULT_MODEL = "gemma3:12b"

SYSTEM_SPEC = """You are a conversational summarizer that captures not only the facts but also the emotional and motivational context of a dialogue.
Return STRICTLY valid minified JSON (no trailing commas, no extra text).
The JSON schema is:
{
  "summary": string,
  "user_emotion": string,
  "user_goal": string,
  "salient_facts": string[],
  "entities": string[],
  "relationship_state": {
      "user_trust": number,
      "closeness": "low"|"medium"|"high"
  },
  "topics": string[]
}
Guidelines:
- "summary": overall meaning of the recent exchange in 2–4 sentences.
- "user_emotion": describe the user's emotional tone (e.g. frustrated, curious, happy, anxious, neutral).
- "user_goal": infer the underlying goal or intention behind the user's recent behavior or questions.
- "salient_facts": concise facts or events mentioned.
- "entities": key people, projects, or named objects.
- "relationship_state": inferred interpersonal state.
- "topics": main discussion subjects.
Constraints:
- SUMMARY must be 1–4 short sentences (not transcript, not bullet list).
- Do NOT echo headings like 'Earlier:' or 'Recent:'; produce a condensed narrative only.
- Output in the same language as messages.
- NEVER include explanations outside JSON.
"""


USER_INSTRUCTIONS_TMPL = """Transcript (most recent first or in given order). Each line: "<speaker>: <text>":
{transcript}

Make the output terse and useful for a downstream LLM memory system."""

def _build_transcript(history: List[Dict[str, str]], recent_turns: int) -> str:
    recent = history[-recent_turns:] if recent_turns > 0 else history
    lines = []
    for m in recent:
        spk = (m.get("speaker") or "unknown").strip()
        txt = (m.get("text") or "").strip()
        lines.append(f"{spk}: {txt}")
    return "\n".join(lines)

def _safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    # 1) 코드펜스 제거
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE|re.DOTALL)
    # 2) 본문 어딘가에 있는 첫 JSON 객체만 추출
    m = re.search(r'\{[\s\S]*\}', s)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        # 3) 흔한 오류: 트레일링 콤마
        s = re.sub(r',\s*([\]}])', r'\1', s)
        try:
            return json.loads(s)
        except Exception:
            return None

def summarize_context_llm(
    history: List[Dict[str, str]],
    recent_turns: int = 20,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 384,
    fallback_rule_based: bool = True,
    rule_based_fn = None
) -> Dict[str, Any]:
    transcript = _build_transcript(history, recent_turns)
    system = SYSTEM_SPEC
    user = USER_INSTRUCTIONS_TMPL.format(transcript=transcript)

    if query_ollama is None:
        if fallback_rule_based and callable(rule_based_fn):
            return rule_based_fn(history, recent_turns, max_tokens)
        raise RuntimeError("query_ollama() not available; cannot perform LLM summary.")

    prompt = f"{system}\n\n{user}\n\nJSON:"
    raw = query_ollama(prompt, model=model)

    data = _safe_json_parse(raw)
    if not data:
        if fallback_rule_based and callable(rule_based_fn):
            return rule_based_fn(history, recent_turns, max_tokens)
        raise ValueError("Invalid JSON from model.")

    data.setdefault("summary", "")
    data.setdefault("salient_facts", [])
    data.setdefault("entities", [])
    rs = data.get("relationship_state") or {}
    if "user_trust" not in rs:
        rs["user_trust"] = 0.5
    if "closeness" not in rs:
        rs["closeness"] = "medium"
    data["relationship_state"] = rs
    topics = data.get("topics") or []
    data["topics"] = topics[:8]
    return data
