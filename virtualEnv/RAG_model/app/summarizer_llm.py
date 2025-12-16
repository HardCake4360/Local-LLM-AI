from typing import List, Dict, Any, Optional
import json, re

try:
    from llmClient import query_ollama
except Exception:
    query_ollama = None

DEFAULT_MODEL = "gemma3:12b"

SYSTEM_SPEC = """너는 대화의 사실뿐만 아니라 감정적·동기적 맥락까지 포착하는 대화 요약기이다.
반드시 유효한 최소화(minified) JSON만 반환해야 하며 (후행 쉼표 금지, JSON 외 텍스트 금지) 다른 설명은 포함하지 않는다.

JSON 스키마는 다음과 같다:
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

가이드라인:
-"summary": 최근 대화의 전체 의미를 2–4문장으로 요약한다.
-"user_emotion": 사용자의 감정적 톤을 설명한다 (예: 좌절, 호기심, 만족, 불안, 중립 등).
-"user_goal": 사용자의 최근 행동이나 질문 이면에 있는 목표나 의도를 추론한다.
-"salient_facts": 언급된 핵심 사실이나 사건을 간결하게 나열한다.
-"entities": 주요 인물, 프로젝트, 고유명사 객체를 포함한다.
-"relationship_state": 사용자와 상대 간의 관계 상태를 추론한다.
-"topics": 주요 논의 주제를 나열한다.

제약 조건:
-SUMMARY는 1–4개의 짧은 문장이어야 하며 (대화 원문이나 목록 형태 금지).
-'Earlier:', 'Recent:' 같은 구분 헤딩을 반복하지 말고 압축된 서술형 요약만 출력한다.
-출력 언어는 입력 메시지와 동일한 언어를 사용한다.
-JSON 외부에 어떠한 설명도 절대 포함하지 않는다.
"""


USER_INSTRUCTIONS_TMPL = """Transcript (가장 최근 발언이 먼저 오거나, 주어진 순서대로 정렬됨).
각 줄의 형식: 
"<speaker>: <text>": {transcript}

출력은 간결하고 핵심 위주로 작성하며, 후속 LLM 메모리 시스템에서 활용하기에 적합해야 한다."""

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
