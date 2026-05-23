#요구사항
"""
pip install flask flask-cors
virtualEnv\\RAG_model\\app\\server.py
"""
import json
import os
import re
import threading
import time
from typing import Any, Dict, List

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
try:
    from openai import OpenAI, OpenAIError
except Exception:
    OpenAI = None
    OpenAIError = Exception

from chatlog import append_log, read_log, read_summary, reset_log, write_summary
from llmClient import build_prompt_v2, query_ollama_stream
from persona_store import load_persona
from retriever import Retriever
from summarizer import summarize_context
from summarizer import summarize_context as summarize_context_rule_based
from summarizer_llm import summarize_context_llm

app = Flask(__name__)
CORS(app)

PDF_NAME = "DatabasePrompt"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "pdfs", PDF_NAME + ".pdf")
INDEX_DIR = os.path.join(DATA_DIR, "index")
INDEX_PATH = os.path.join(INDEX_DIR, "world")
WORLD_DIR = os.path.join(DATA_DIR, "world")
CHATLOG_DIR = os.path.join(DATA_DIR, "chatlogs")
OLLAMA_MANIFEST_DIR = os.path.join(os.path.expanduser("~"), ".ollama", "models", "manifests", "registry.ollama.ai", "library")
ACTIVE_MODEL = "gemma3:12b"
LLM_PROVIDER = "ollama"
OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_REASONING_EFFORT = "none"
APP_ENV_PATH = os.path.join(BASE_DIR, "apiKey.env")

retriever = Retriever()
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CHATLOG_DIR, exist_ok=True)


def _load_env_file(path: str) -> None:
    if not os.path.isfile(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as error:
        print(f"[WARN] env file load failed path={path}: {error}")


_load_env_file(APP_ENV_PATH)


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_reply_text(text: Any) -> str:
    if text is None:
        return "..."

    cleaned = str(text)
    cleaned = cleaned.replace("<|eot_id|>", " ")
    cleaned = cleaned.replace("</s>", " ")
    cleaned = cleaned.replace("<end_of_turn>", " ")
    cleaned = re.sub(r"<\|.*?\|>", " ", cleaned)
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned or "..."


def _discover_installed_models() -> List[str]:
    models: List[str] = []
    if not os.path.isdir(OLLAMA_MANIFEST_DIR):
        return models

    for model_name in sorted(os.listdir(OLLAMA_MANIFEST_DIR)):
        model_dir = os.path.join(OLLAMA_MANIFEST_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue

        tags = []
        for tag in sorted(os.listdir(model_dir)):
            tag_path = os.path.join(model_dir, tag)
            if os.path.isfile(tag_path):
                tags.append(tag)

        if not tags:
            continue

        for tag in tags:
            models.append(f"{model_name}:{tag}")

    return models


def _is_openai_selected() -> bool:
    return LLM_PROVIDER == "openai"


def _require_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK is not installed. Install it with: pip install openai")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    return OpenAI()


def _get_openai_error_detail(error: Exception) -> Dict[str, str]:
    status_code = getattr(error, "status_code", "")
    error_code = getattr(error, "code", "")
    error_type = getattr(error, "type", "")
    message = str(error)

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body_error = body.get("error")
        if isinstance(body_error, dict):
            error_code = error_code or body_error.get("code", "")
            error_type = error_type or body_error.get("type", "")
            message = body_error.get("message", message)

    if not error_code and "insufficient_quota" in message:
        error_code = "insufficient_quota"
    if not error_type and "insufficient_quota" in message:
        error_type = "insufficient_quota"

    return {
        "status_code": str(status_code or "unknown"),
        "code": str(error_code or "unknown"),
        "type": str(error_type or error.__class__.__name__),
        "message": _safe_reply_text(message),
    }


def _openai_error_hint(detail: Dict[str, str]) -> str:
    code = detail.get("code", "")
    error_type = detail.get("type", "")
    status_code = detail.get("status_code", "")

    if code == "insufficient_quota" or error_type == "insufficient_quota":
        return "API key was loaded, but the OpenAI project has no available quota or has hit its monthly spend limit. Check Platform billing, credits, project, and limits."

    if status_code == "401":
        return "Authentication failed. Check that apiKey.env contains the active key for the intended OpenAI project."

    if status_code == "403":
        return "Permission denied. Check model access, project permissions, organization, or IP allowlist settings."

    if status_code == "429":
        return "Rate limit reached. Slow requests down or check the project's model rate limits."

    if "OPENAI_API_KEY is not configured" in detail.get("message", ""):
        return "OPENAI_API_KEY was not found in the process environment. Check APP_ENV_PATH and apiKey.env."

    if "OpenAI SDK is not installed" in detail.get("message", ""):
        return "OpenAI SDK is missing from the active Python environment. Install it in RagEnv."

    return "OpenAI request failed. Check the detailed error fields above."


def _log_llm_error(operation: str, error: Exception) -> None:
    if LLM_PROVIDER != "openai":
        print(
            f"[ERROR][LLM] operation={operation} provider={LLM_PROVIDER} "
            f"ollama_model={ACTIVE_MODEL} error={error}"
        )
        return

    detail = _get_openai_error_detail(error)
    print(
        "[ERROR][LLM][OPENAI] "
        f"operation={operation} model={OPENAI_MODEL} "
        f"status={detail['status_code']} code={detail['code']} type={detail['type']}"
    )
    print(f"[ERROR][LLM][OPENAI] message={detail['message']}")
    print(f"[ERROR][LLM][OPENAI] hint={_openai_error_hint(detail)}")


def _query_openai_once(prompt: str) -> str:
    client = _require_openai_client()
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
    )
    return _safe_reply_text(getattr(response, "output_text", ""))


def _query_openai_stream(prompt: str):
    client = _require_openai_client()
    stream = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        reasoning={"effort": OPENAI_REASONING_EFFORT},
        stream=True,
    )

    for event in stream:
        if getattr(event, "type", "") == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                yield delta
        elif getattr(event, "type", "") == "error":
            error = getattr(event, "error", None) or event
            raise RuntimeError(f"OpenAI stream error: {error}")


def _query_active_llm_stream(prompt: str, local_model: str = ""):
    if _is_openai_selected():
        try:
            yield from _query_openai_stream(prompt)
        except Exception as error:
            _log_llm_error("stream", error)
            raise
        return

    model = local_model or ACTIVE_MODEL
    try:
        yield from query_ollama_stream(prompt, model)
    except Exception as error:
        _log_llm_error("stream", error)
        raise


def _validate_openai_startup() -> bool:
    print(f"[OPENAI] startup validation model={OPENAI_MODEL}")
    try:
        reply = _query_openai_once("Reply with exactly: OK")
        print(f"[OPENAI] startup validation succeeded: {reply}")
        return True
    except Exception as error:
        _log_llm_error("startup_validation", error)
        print("[OPENAI] startup validation failed. The server will still start; OpenAI replies will fail until the logged issue is fixed.")
        return False


def _select_startup_model() -> str:
    global ACTIVE_MODEL, LLM_PROVIDER

    models = _discover_installed_models()
    options = [{"provider": "ollama", "model": model_name, "label": model_name} for model_name in models]
    options.append({
        "provider": "openai",
        "model": OPENAI_MODEL,
        "label": f"OpenAI API ({OPENAI_MODEL})",
    })

    print("\n[MODEL] 사용 가능한 LLM 옵션")
    for index, option in enumerate(options, start=1):
        marker = " (default)" if option["provider"] == "ollama" and option["model"] == ACTIVE_MODEL else ""
        print(f"{index}. {option['label']}{marker}")

    while True:
        try:
            raw = input(f"사용할 LLM 번호를 입력하세요 [1-{len(options)}], 엔터 시 기본값: ").strip()
        except EOFError:
            raw = ""

        if not raw:
            selected_option = next(
                (option for option in options if option["provider"] == "ollama" and option["model"] == ACTIVE_MODEL),
                options[0],
            )
            break

        if raw.isdigit():
            selected_index = int(raw)
            if 1 <= selected_index <= len(options):
                selected_option = options[selected_index - 1]
                break

        print("잘못된 입력입니다. 숫자를 다시 입력해주세요.")

    LLM_PROVIDER = selected_option["provider"]
    if LLM_PROVIDER == "ollama":
        ACTIVE_MODEL = selected_option["model"]
        print(f"[MODEL] 선택된 로컬 모델: {ACTIVE_MODEL}")
    else:
        print(f"[MODEL] 선택된 OpenAI API 모델: {OPENAI_MODEL}")
        _validate_openai_startup()

    return ACTIVE_MODEL


def _user_log_path(user_id: str) -> str:
    safe_user = (user_id or "anonymous").strip() or "anonymous"
    return os.path.join(CHATLOG_DIR, f"{safe_user}.jsonl")


def _summary_path(user_id: str, persona_key: str) -> str:
    safe_user = (user_id or "anonymous").strip() or "anonymous"
    safe_persona = (persona_key or "assistant").strip() or "assistant"
    return os.path.join(CHATLOG_DIR, f"{safe_user}__{safe_persona}.summary.json")


def _conversation_scope(persona_key: str, fallback: str = "assistant") -> str:
    return (persona_key or "").strip() or fallback


def _append_scoped_log(user_id: str, persona_key: str, speaker: str, text: str) -> None:
    rec = {
        "ts": time.time(),
        "speaker": speaker,
        "text": text,
        "personaKey": _conversation_scope(persona_key),
    }
    with open(_user_log_path(user_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_scoped_log(user_id: str, persona_key: str, max_items: int = 200) -> List[Dict[str, Any]]:
    path = _user_log_path(user_id)
    if not os.path.exists(path):
        return []

    scope = _conversation_scope(persona_key)
    lines: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if (rec.get("personaKey") or "").strip() != scope:
                continue
            lines.append(rec)

    return lines[-max_items:]


def _write_scoped_summary(user_id: str, persona_key: str, summary: Dict[str, Any]) -> None:
    with open(_summary_path(user_id, persona_key), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)


def _read_scoped_summary(user_id: str, persona_key: str) -> Dict[str, Any] | None:
    path = _summary_path(user_id, persona_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _reset_scoped_log(user_id: str, persona_key: str) -> None:
    path = _user_log_path(user_id)
    if os.path.exists(path):
        scope = _conversation_scope(persona_key)
        kept: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if (rec.get("personaKey") or "").strip() != scope:
                    kept.append(rec)

        with open(path, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_path = _summary_path(user_id, persona_key)
    if os.path.exists(summary_path):
        os.remove(summary_path)


def _default_interrogation_state() -> Dict[str, Any]:
    return {
        "tell": 0.0,
        "affect": {
            "interest": 0.0,
            "attitude": 0.0,
        },
        "patience": 100,
    }


def _default_conversation_state() -> Dict[str, Any]:
    return {
        "affect": {
            "interest": 0.0,
            "attitude": 0.0,
        },
        "patience": 100,
    }


def _default_profile() -> Dict[str, Any]:
    return {
        "baseInterest": 0.0,
        "baseAttitude": 0.0,
        "basePatience": 100,
        "actionWeights": [
            {
                "actionType": "Talk",
                "tellDelta": 0.05,
                "interestDelta": 0.01,
                "attitudeDelta": 0.0,
                "patienceCost": 4,
            },
            {
                "actionType": "AskTopic",
                "tellDelta": 0.20,
                "interestDelta": 0.04,
                "attitudeDelta": 0.0,
                "patienceCost": 6,
            },
            {
                "actionType": "PresentEvidence",
                "tellDelta": 0.42,
                "interestDelta": 0.06,
                "attitudeDelta": -0.08,
                "patienceCost": 10,
            },
        ],
        "pressureKeywordRules": [],
        "sensitiveKeywordRules": [],
        "topicRules": [],
        "evidenceRules": [],
    }


def _normalize_affect(affect: Dict[str, Any]) -> Dict[str, float]:
    affect = affect or {}
    return {
        "interest": round(_clamp_signed(_to_float(affect.get("interest"), 0.0)), 2),
        "attitude": round(_clamp_signed(_to_float(affect.get("attitude"), 0.0)), 2),
    }


def _normalize_interrogation_state(state: Dict[str, Any]) -> Dict[str, Any]:
    state = state or {}
    return {
        "tell": round(_clamp(_to_float(state.get("tell"), 0.0), 0.0, 1.0), 2),
        "affect": _normalize_affect(state.get("affect")),
        "patience": max(0, min(100, _to_int(state.get("patience"), 100))),
    }


def _normalize_conversation_state(state: Dict[str, Any]) -> Dict[str, Any]:
    state = state or {}
    return {
        "affect": _normalize_affect(state.get("affect")),
        "patience": max(0, min(100, _to_int(state.get("patience"), 100))),
    }


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _tell_log_band(tell_value: float) -> str:
    tell_value = _clamp(_to_float(tell_value, 0.0), 0.0, 1.0)
    if tell_value < 0.25:
        return "Stable"
    if tell_value < 0.50:
        return "Guarded"
    if tell_value < 0.75:
        return "Shaken"
    return "Disturbed"


def _build_tell_classifier_prompt(
    question_text: str,
    persona: Dict[str, Any],
) -> str:
    personality = persona.get("personality") or {}
    affective_rules = persona.get("affective_rules") or {}
    on_action = affective_rules.get("on_action") or {}

    defense_mechanisms = personality.get("defense_mechanisms") or []
    emotional_baseline = personality.get("emotional_baseline") or ""
    on_action_json = json.dumps(on_action, ensure_ascii=False, indent=2)
    defense_json = json.dumps(defense_mechanisms, ensure_ascii=False, indent=2)

    return f"""
너는 추리 게임 NPC 반응 분석기다.
입력된 질문과 페르소나 정보만 보고, 이번 턴에 한정된 tell 수치를 계산한다.

[tell 정의]
- tell은 NPC가 이번 턴 질문에 얼마나 동요했는지를 나타내는 0~1 수치다.
- tell은 거짓말 여부를 직접 뜻하지 않는다.
- 높은 tell은 예상치 못한 주제, 방어기제 자극, 감정적 흔들림, 긴장, 회피 충동을 의미할 수 있다.
- tell은 이번 턴 한정 반응이며 누적값이 아니다.

[평가 입력]
- 질문 내용:
{_safe_reply_text(question_text)}

- defense_mechanisms:
{defense_json}

- emotional_baseline:
{emotional_baseline}

- on_action:
{on_action_json}

[평가 지침]
- 질문을 on_action 범주 중 가장 가까운 행위로 분류하라. 필요하면 복수 범주를 참고하되 primaryAction은 하나만 선택한다.
- defense_mechanisms가 강하게 자극될수록 tell을 높인다.
- emotional_baseline이 차분하고 억제적인 경우, 겉으로 큰 감정 폭발이 없어도 내부 동요가 있을 수 있다.
- tell은 "동요"이지 "거짓말"이 아니다.
- 출력 tell은 0~1 사이 float 하나만 사용한다.

[로그용 구간 규칙]
- 0.00~0.24: Stable
- 0.25~0.49: Guarded
- 0.50~0.74: Shaken
- 0.75~1.00: Disturbed

[출력 형식]
반드시 JSON 객체 하나만 출력하라. 설명문, 코드블록, 마크다운 금지.
형식:
{{
  "primaryAction": "분류한 행위명",
  "tell": 0.0,
  "band": "Stable|Guarded|Shaken|Disturbed",
  "reason": "한 문장 요약"
}}
""".strip()


def _derive_tell_llm(question_text: str, persona: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_tell_classifier_prompt(question_text, persona)
    raw_chunks: List[str] = []

    for chunk in _query_active_llm_stream(prompt):
        raw_chunks.append(chunk)

    raw_text = _safe_reply_text("".join(raw_chunks))
    parsed = _extract_json_object(raw_text)

    if not parsed:
        return {
            "tell": 0.0,
            "band": "Stable",
            "primaryAction": "unknown",
            "reason": "LLM tell classifier returned invalid JSON.",
            "raw": raw_text,
        }

    tell_value = round(_clamp(_to_float(parsed.get("tell"), 0.0), 0.0, 1.0), 2)
    band = parsed.get("band") or _tell_log_band(tell_value)
    primary_action = (parsed.get("primaryAction") or "unknown").strip() or "unknown"
    reason = _safe_reply_text(parsed.get("reason") or "")

    return {
        "tell": tell_value,
        "band": band,
        "primaryAction": primary_action,
        "reason": reason,
        "raw": raw_text,
    }


def _normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    merged = _default_profile()
    profile = profile or {}

    merged["baseInterest"] = _clamp_signed(_to_float(profile.get("baseInterest"), merged["baseInterest"]))
    merged["baseAttitude"] = _clamp_signed(_to_float(profile.get("baseAttitude"), merged["baseAttitude"]))
    merged["basePatience"] = max(0, min(100, _to_int(profile.get("basePatience"), merged["basePatience"])))
    merged["actionWeights"] = profile.get("actionWeights") or merged["actionWeights"]
    merged["pressureKeywordRules"] = profile.get("pressureKeywordRules") or []
    merged["sensitiveKeywordRules"] = profile.get("sensitiveKeywordRules") or []
    merged["topicRules"] = profile.get("topicRules") or []
    merged["evidenceRules"] = profile.get("evidenceRules") or []
    return merged


def _find_action_weight(profile: Dict[str, Any], action_type: str) -> Dict[str, Any]:
    for rule in profile.get("actionWeights", []):
        if (rule.get("actionType") or "").strip() == (action_type or "").strip():
            return rule
    return {
        "tellDelta": 0.0,
        "interestDelta": 0.0,
        "attitudeDelta": 0.0,
        "patienceCost": 4,
    }


def _apply_keyword_rules(text: str, rules: List[Dict[str, Any]], state: Dict[str, Any]) -> int:
    patience_delta = 0
    text = text or ""
    for rule in rules or []:
        pattern = rule.get("pattern") or ""
        if not pattern:
            continue
        if re.search(pattern, text, re.IGNORECASE):
            state["affect"]["interest"] += _to_float(rule.get("interestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("attitudeDelta"), 0.0)
            patience_delta += _to_int(rule.get("patienceCost"), 0)
    return patience_delta


def _apply_topic_rules(topic_id: str, unlocked_topic_ids: set, profile: Dict[str, Any], state: Dict[str, Any]) -> None:
    if not topic_id:
        return

    for rule in profile.get("topicRules", []):
        if (rule.get("topicId") or "") != topic_id:
            continue

        if topic_id in unlocked_topic_ids:
            state["affect"]["interest"] += _to_float(rule.get("knownInterestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("knownAttitudeDelta"), 0.0)
        else:
            state["affect"]["interest"] += _to_float(rule.get("unknownInterestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("unknownAttitudeDelta"), 0.0)


def _apply_evidence_rules(evidence_id: str, discovered_evidence_ids: set, profile: Dict[str, Any], state: Dict[str, Any]) -> None:
    if not evidence_id:
        return

    for rule in profile.get("evidenceRules", []):
        if (rule.get("evidenceId") or "") != evidence_id:
            continue

        if evidence_id in discovered_evidence_ids:
            state["affect"]["interest"] += _to_float(rule.get("discoveredInterestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("discoveredAttitudeDelta"), 0.0)
        else:
            state["affect"]["interest"] += _to_float(rule.get("undiscoveredInterestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("undiscoveredAttitudeDelta"), 0.0)


def _derive_interrogation_state(
    action_type: str,
    interaction: Dict[str, Any],
    scene_state: Dict[str, Any],
    npc_local_state: Dict[str, Any],
    conversation_context: Dict[str, Any],
    profile: Dict[str, Any],
    persona: Dict[str, Any],
) -> Dict[str, Any]:
    profile = _normalize_profile(profile)
    last_known_affect = _normalize_affect((npc_local_state or {}).get("lastKnownAffect"))
    last_known_patience = max(0, min(100, _to_int((npc_local_state or {}).get("lastKnownPatience"), profile.get("basePatience", 100))))

    state = {
        "tell": 0.0,
        "affect": {
            "interest": last_known_affect["interest"] if npc_local_state else profile.get("baseInterest", 0.0),
            "attitude": last_known_affect["attitude"] if npc_local_state else profile.get("baseAttitude", 0.0),
        },
        "patience": last_known_patience,
    }

    action_weight = _find_action_weight(profile, action_type)
    state["affect"]["interest"] += _to_float(action_weight.get("interestDelta"), 0.0)
    state["affect"]["attitude"] += _to_float(action_weight.get("attitudeDelta"), 0.0)
    patience_cost = _to_int(action_weight.get("patienceCost"), 4)

    player_intent_text = (interaction or {}).get("playerIntentText") or ""
    topic_id = (interaction or {}).get("topicId")
    evidence_id = (interaction or {}).get("evidenceId")
    unlocked_topic_ids = set((scene_state or {}).get("unlockedTopicIds") or [])
    discovered_evidence_ids = set((scene_state or {}).get("discoveredEvidenceIds") or [])
    recent_exchanges = (conversation_context or {}).get("recentExchanges") or []
    tell_analysis = _derive_tell_llm(player_intent_text, persona or {})

    state["tell"] = tell_analysis["tell"]

    patience_cost += _apply_keyword_rules(player_intent_text, profile.get("pressureKeywordRules"), state)
    patience_cost += _apply_keyword_rules(player_intent_text, profile.get("sensitiveKeywordRules"), state)
    _apply_topic_rules(topic_id, unlocked_topic_ids, profile, state)
    _apply_evidence_rules(evidence_id, discovered_evidence_ids, profile, state)

    repeated_question = False
    normalized_intent = _safe_reply_text(player_intent_text)
    for exchange in recent_exchanges[-4:]:
        if exchange.get("speaker") != "player":
            continue
        if _safe_reply_text(exchange.get("text")) == normalized_intent:
            repeated_question = True
            break

    if repeated_question:
        state["affect"]["interest"] -= 0.12
        state["affect"]["attitude"] -= 0.06
        patience_cost += 6

    if state["affect"]["interest"] >= 0.4:
        state["affect"]["attitude"] += 0.05
        patience_cost -= 1
    elif state["affect"]["interest"] <= -0.3:
        state["affect"]["attitude"] -= 0.06
        patience_cost += 3

    if state["affect"]["attitude"] >= 0.4:
        patience_cost -= 2
    elif state["affect"]["attitude"] <= -0.4:
        patience_cost += 4

    if state["patience"] <= 35:
        state["affect"]["attitude"] -= 0.04
        patience_cost += 2

    state["patience"] = max(0, min(100, state["patience"] - max(1, patience_cost)))
    state["tell"] = round(_clamp(state["tell"], 0.0, 1.0), 2)
    state["affect"]["interest"] = round(_clamp_signed(state["affect"]["interest"]), 2)
    state["affect"]["attitude"] = round(_clamp_signed(state["affect"]["attitude"]), 2)
    state["tellMeta"] = {
        "band": _tell_log_band(state["tell"]),
        "primaryAction": tell_analysis["primaryAction"],
        "reason": tell_analysis["reason"],
    }
    return state


def _derive_conversation_state(
    action_type: str,
    interaction: Dict[str, Any],
    scene_state: Dict[str, Any],
    npc_local_state: Dict[str, Any],
    conversation_context: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    profile = _normalize_profile(profile)
    last_known_affect = _normalize_affect((npc_local_state or {}).get("lastKnownAffect"))
    last_known_patience = max(0, min(100, _to_int((npc_local_state or {}).get("lastKnownPatience"), profile.get("basePatience", 100))))

    state = {
        "affect": {
            "interest": last_known_affect["interest"] if npc_local_state else profile.get("baseInterest", 0.0),
            "attitude": last_known_affect["attitude"] if npc_local_state else profile.get("baseAttitude", 0.0),
        },
        "patience": last_known_patience,
    }

    action_weight = _find_action_weight(profile, action_type)
    state["affect"]["interest"] += _to_float(action_weight.get("interestDelta"), 0.0)
    state["affect"]["attitude"] += _to_float(action_weight.get("attitudeDelta"), 0.0)
    patience_cost = _to_int(action_weight.get("patienceCost"), 4)

    player_intent_text = (interaction or {}).get("playerIntentText") or ""
    topic_id = (interaction or {}).get("topicId")
    evidence_id = (interaction or {}).get("evidenceId")
    unlocked_topic_ids = set((scene_state or {}).get("unlockedTopicIds") or [])
    discovered_evidence_ids = set((scene_state or {}).get("discoveredEvidenceIds") or [])
    recent_exchanges = (conversation_context or {}).get("recentExchanges") or []

    patience_cost += _apply_keyword_rules(player_intent_text, profile.get("pressureKeywordRules"), state)
    patience_cost += _apply_keyword_rules(player_intent_text, profile.get("sensitiveKeywordRules"), state)
    _apply_topic_rules(topic_id, unlocked_topic_ids, profile, state)
    _apply_evidence_rules(evidence_id, discovered_evidence_ids, profile, state)

    repeated_question = False
    normalized_intent = _safe_reply_text(player_intent_text)
    for exchange in recent_exchanges[-4:]:
        if exchange.get("speaker") != "player":
            continue
        if _safe_reply_text(exchange.get("text")) == normalized_intent:
            repeated_question = True
            break

    if repeated_question:
        state["affect"]["interest"] -= 0.12
        state["affect"]["attitude"] -= 0.06
        patience_cost += 6

    if state["affect"]["interest"] >= 0.4:
        state["affect"]["attitude"] += 0.05
        patience_cost -= 1
    elif state["affect"]["interest"] <= -0.3:
        state["affect"]["attitude"] -= 0.06
        patience_cost += 3

    if state["affect"]["attitude"] >= 0.4:
        patience_cost -= 2
    elif state["affect"]["attitude"] <= -0.4:
        patience_cost += 4

    if state["patience"] <= 35:
        state["affect"]["attitude"] -= 0.04
        patience_cost += 2

    state["patience"] = max(0, min(100, state["patience"] - max(1, patience_cost)))
    state["affect"]["interest"] = round(_clamp_signed(state["affect"]["interest"]), 2)
    state["affect"]["attitude"] = round(_clamp_signed(state["affect"]["attitude"]), 2)
    return state


def _interrogation_state_to_presentation_hints(state: Dict[str, Any]) -> Dict[str, Any]:
    state = _normalize_interrogation_state(state)
    tell = state["tell"]
    interest = state["affect"]["interest"]
    attitude = state["affect"]["attitude"]
    patience = state["patience"]

    if tell >= 0.7:
        animation = "tense_idle"
    elif tell >= 0.35 or patience <= 30:
        animation = "guarded_idle"
    else:
        animation = "calm_idle"

    if attitude <= -0.4 or patience <= 20:
        voice_tone = "hostile"
    elif attitude >= 0.35 and interest >= 0.2:
        voice_tone = "cooperative"
    else:
        voice_tone = "guarded"

    return {
        "animation": animation,
        "voiceTone": voice_tone,
        "uiNoiseLevel": round(tell, 2),
    }


def _conversation_state_to_presentation_hints(state: Dict[str, Any]) -> Dict[str, Any]:
    state = _normalize_conversation_state(state)
    interest = state["affect"]["interest"]
    attitude = state["affect"]["attitude"]
    patience = state["patience"]

    if patience <= 30:
        animation = "guarded_idle"
    elif interest >= 0.35 and attitude >= 0.2:
        animation = "calm_idle"
    else:
        animation = "guarded_idle"

    if attitude <= -0.4 or patience <= 20:
        voice_tone = "hostile"
    elif attitude >= 0.35 and interest >= 0.2:
        voice_tone = "cooperative"
    else:
        voice_tone = "guarded"

    return {
        "animation": animation,
        "voiceTone": voice_tone,
        "uiNoiseLevel": 0.0,
    }


def _build_statement_id(npc_id: str, npc_state: Dict[str, Any]) -> str:
    count = int(_to_float((npc_state or {}).get("conversationCount"), 0)) + 1
    safe_npc = re.sub(r"[^a-zA-Z0-9_]+", "_", npc_id or "npc").strip("_") or "npc"
    return f"{safe_npc}_stmt_{count:03d}"


def _extract_unlock_topics(interaction: Dict[str, Any], reply_text: str) -> List[str]:
    unlocks: List[str] = []
    text = reply_text or ""
    action_type = (interaction or {}).get("actionType") or "Talk"
    evidence_id = (interaction or {}).get("evidenceId")

    # TODO: 빠른 구현용 서버 하드코딩. 프로젝트 종료 전 클라이언트 데이터로 이관 필요.
    keyword_rules = [
        (r"노이즈|잡음|화이트\s*노이즈", "noise_time"),
        (r"보청기", "hearing_aid_amplification_possibility"),
        (r"CRT|브라운관", "crt_signal_origin"),
    ]

    for pattern, topic_id in keyword_rules:
        if re.search(pattern, text, re.IGNORECASE) and topic_id not in unlocks:
            unlocks.append(topic_id)

    if action_type == "PresentEvidence" and evidence_id:
        followup_id = f"{evidence_id}_followup"
        if followup_id not in unlocks:
            unlocks.append(followup_id)

    return unlocks


def _build_investigation_prompt(
    payload: Dict[str, Any],
    persona: Dict[str, Any],
    interrogation_state: Dict[str, Any],
) -> str:
    scene_id = payload.get("sceneId") or "unknown_scene"
    npc_id = payload.get("npcId") or "unknown_npc"
    phase = payload.get("phase") or "investigation"
    interaction = payload.get("interaction") or {}
    scene_state = payload.get("sceneState") or {}
    npc_local_state = payload.get("npcLocalState") or {}
    conversation_context = payload.get("conversationContext") or {}

    action_type = interaction.get("actionType") or "Talk"
    player_intent_text = interaction.get("playerIntentText") or ""
    topic_id = interaction.get("topicId")
    evidence_id = interaction.get("evidenceId")

    recent_exchanges = conversation_context.get("recentExchanges") or []
    recent_lines = []
    for ex in recent_exchanges[-6:]:
        speaker = ex.get("speaker", "unknown")
        text = _safe_reply_text(ex.get("text", ""))
        if text:
            recent_lines.append(f"- {speaker}: {text}")

    persona_json = json.dumps(persona or {}, ensure_ascii=False, indent=2)
    scene_state_json = json.dumps(scene_state, ensure_ascii=False, indent=2)
    npc_state_json = json.dumps(npc_local_state, ensure_ascii=False, indent=2)
    interrogation_state_json = json.dumps(_normalize_interrogation_state(interrogation_state), ensure_ascii=False, indent=2)

    prompt = f"""
너는 한국어 추리 게임 속 NPC다. 지금부터 조사 파트 전용 응답만 생성한다.

[페르소나 데이터]
{persona_json}

[현재 조사 정보]
- sceneId: {scene_id}
- phase: {phase}
- npcId: {npc_id}
- actionType: {action_type}
- playerIntentText: {_safe_reply_text(player_intent_text)}
- topicId: {topic_id}
- evidenceId: {evidence_id}

[사건 상태]
{scene_state_json}

[NPC 로컬 상태]
{npc_state_json}

[이번 질문에 대한 심문 상태]
{interrogation_state_json}

[최근 대화]
{os.linesep.join(recent_lines) if recent_lines else '- 없음'}

[응답 원칙]
- 반드시 한국어로만 대답한다.
- 아래 심문 상태를 보고 답변의 길이, 협조성, 방어성, 동요 정도를 조절한다.
- tell이 높을수록 머뭇거림, 회피, 자기수정, 말꼬임, 제한적인 모순 가능성을 보인다.
- affect.interest가 높을수록 답변이 자세해지고, 낮을수록 짧고 건조해진다.
- affect.attitude가 낮을수록 차갑고 방어적이며, 높을수록 협조적이다.
- patience가 낮을수록 짧고 예민한 답변을 한다. patience가 매우 낮으면 더 이상 길게 설명하지 않으려 한다.
- 사건 상태와 최근 대화와 모순되지 않도록 한다.
- 시스템 설명, 상태 수치, JSON, 메타 발언은 절대 출력하지 않는다.

이제 NPC의 실제 대사만 출력해라.
""".strip()
    return prompt


def _normalize_attached_item_directive(interaction: Dict[str, Any]) -> Dict[str, Any]:
    directive = (interaction or {}).get("attachedItemDirective") or {}
    return {
        "itemId": str(directive.get("itemId") or ""),
        "itemCategory": str(directive.get("itemCategory") or ""),
        "itemDisplayName": str(directive.get("itemDisplayName") or ""),
        "itemPrompt": _safe_reply_text(directive.get("itemPrompt") or ""),
        "attitudeThreshold": round(_to_float(directive.get("attitudeThreshold"), 0.0), 3),
        "currentAttitude": round(_to_float(directive.get("currentAttitude"), 0.0), 3),
        "thresholdSatisfied": bool(directive.get("thresholdSatisfied")),
        "responseDirectivePrompt": _safe_reply_text(directive.get("responseDirectivePrompt") or ""),
        "bypassWorldContextRag": bool(directive.get("bypassWorldContextRag")),
        "followingItemId": str(directive.get("followingItemId") or ""),
        "followingItemCategory": str(directive.get("followingItemCategory") or ""),
    }


def _format_attached_item_directive(attached_item_directive: Dict[str, Any]) -> str:
    if not attached_item_directive or not attached_item_directive.get("responseDirectivePrompt"):
        return "- 없음"

    lines = [
        f"- itemId: {attached_item_directive.get('itemId', '')}",
        f"- itemCategory: {attached_item_directive.get('itemCategory', '')}",
        f"- itemDisplayName: {attached_item_directive.get('itemDisplayName', '')}",
        f"- currentAttitude: {attached_item_directive.get('currentAttitude', 0.0)}",
        f"- attitudeThreshold: {attached_item_directive.get('attitudeThreshold', 0.0)}",
        f"- thresholdSatisfied: {attached_item_directive.get('thresholdSatisfied', False)}",
        f"- bypassWorldContextRag: {attached_item_directive.get('bypassWorldContextRag', False)}",
        f"- followingItemId: {attached_item_directive.get('followingItemId', '')}",
        f"- responseDirectivePrompt: {attached_item_directive.get('responseDirectivePrompt', '')}",
    ]
    return os.linesep.join(lines)


def _build_reply_prompt_general(
    payload: Dict[str, Any],
    persona: Dict[str, Any],
    conversation_state: Dict[str, Any],
    summary_context: Dict[str, Any] | None,
    world_context_chunks: List[str],
) -> str:
    scene_id = payload.get("sceneId") or "unknown_scene"
    npc_id = payload.get("npcId") or "unknown_npc"
    phase = payload.get("phase") or "investigation"
    interaction = payload.get("interaction") or {}
    scene_state = payload.get("sceneState") or {}
    npc_local_state = payload.get("npcLocalState") or {}
    conversation_context = payload.get("conversationContext") or {}

    action_type = interaction.get("actionType") or "Talk"
    player_intent_text = interaction.get("playerIntentText") or ""
    topic_id = interaction.get("topicId")
    evidence_id = interaction.get("evidenceId")

    recent_exchanges = conversation_context.get("recentExchanges") or []
    recent_lines = []
    for ex in recent_exchanges[-6:]:
        speaker = ex.get("speaker", "unknown")
        text = _safe_reply_text(ex.get("text", ""))
        if text:
            recent_lines.append(f"- {speaker}: {text}")

    persona_json = json.dumps(persona or {}, ensure_ascii=False, indent=2)
    scene_state_json = json.dumps(scene_state, ensure_ascii=False, indent=2)
    npc_state_json = json.dumps(npc_local_state, ensure_ascii=False, indent=2)
    conversation_state_json = json.dumps(_normalize_conversation_state(conversation_state), ensure_ascii=False, indent=2)
    summary_text = _format_summary_context(summary_context)
    world_context_text = _format_world_context(world_context_chunks)

    prompt = f"""
너는 한국어 추리 게임 속 NPC다. 지금부터 조사 파트 전용 응답만 생성한다.

[페르소나 데이터]
{persona_json}

[현재 조사 정보]
- sceneId: {scene_id}
- phase: {phase}
- npcId: {npc_id}
- actionType: {action_type}
- playerIntentText: {_safe_reply_text(player_intent_text)}
- topicId: {topic_id}
- evidenceId: {evidence_id}

[사건 상태]
{scene_state_json}

[NPC 로컬 상태]
{npc_state_json}

[이번 질문에 대한 대화 상태]
{conversation_state_json}

[대화 요약]
{summary_text}

[참조 world context]
{world_context_text}

[최근 대화]
{os.linesep.join(recent_lines) if recent_lines else '- 없음'}

[응답 원칙]
- 반드시 한국어로만 대답한다.
- affect.interest가 높을수록 답변이 자세해지고, 낮을수록 짧고 건조해진다.
- affect.attitude가 낮을수록 차갑고 방어적이며, 높을수록 협조적이다.
- patience가 낮을수록 짧고 예민한 답변을 한다. patience가 매우 낮으면 더 이상 길게 설명하지 않으려 한다.
- 사건 상태와 최근 대화와 모순되지 않도록 한다.
- 시스템 설명, 상태 수치, JSON, 메타 발언은 절대 출력하지 않는다.

이제 NPC의 실제 대사만 출력해라.
""".strip()
    return prompt


def _build_reply_prompt_attached(
    payload: Dict[str, Any],
    persona: Dict[str, Any],
    conversation_state: Dict[str, Any],
    attached_item_directive: Dict[str, Any],
) -> str:
    scene_id = payload.get("sceneId") or "unknown_scene"
    npc_id = payload.get("npcId") or "unknown_npc"
    phase = payload.get("phase") or "investigation"
    interaction = payload.get("interaction") or {}
    scene_state = payload.get("sceneState") or {}
    npc_local_state = payload.get("npcLocalState") or {}
    conversation_context = payload.get("conversationContext") or {}

    action_type = interaction.get("actionType") or "Talk"
    player_intent_text = interaction.get("playerIntentText") or ""
    topic_id = interaction.get("topicId")
    evidence_id = interaction.get("evidenceId")

    recent_exchanges = conversation_context.get("recentExchanges") or []
    recent_lines = []
    for ex in recent_exchanges[-6:]:
        speaker = ex.get("speaker", "unknown")
        text = _safe_reply_text(ex.get("text", ""))
        if text:
            recent_lines.append(f"- {speaker}: {text}")

    persona_json = json.dumps(persona or {}, ensure_ascii=False, indent=2)
    scene_state_json = json.dumps(scene_state, ensure_ascii=False, indent=2)
    npc_state_json = json.dumps(npc_local_state, ensure_ascii=False, indent=2)
    conversation_state_json = json.dumps(_normalize_conversation_state(conversation_state), ensure_ascii=False, indent=2)
    attached_item_text = _format_attached_item_directive(attached_item_directive)

    prompt = f"""
너는 한국어 추리 게임 속 NPC다. 지금부터 첨부된 조사 정보에 대한 응답만 생성한다.

[페르소나 데이터]
{persona_json}

[현재 조사 정보]
- sceneId: {scene_id}
- phase: {phase}
- npcId: {npc_id}
- actionType: {action_type}
- playerIntentText: {_safe_reply_text(player_intent_text)}
- topicId: {topic_id}
- evidenceId: {evidence_id}

[사건 상태]
{scene_state_json}

[NPC 로컬 상태]
{npc_state_json}

[이번 질문에 대한 대화 상태]
{conversation_state_json}

[첨부된 조사 정보 지시]
{attached_item_text}

[최근 대화]
{os.linesep.join(recent_lines) if recent_lines else '- 없음'}

[응답 원칙]
- 반드시 한국어로만 대답한다.
- affect.interest가 높을수록 답변이 자세해지고, 낮을수록 짧고 건조해진다.
- affect.attitude가 낮을수록 차갑고 방어적이며, 높을수록 협조적이다.
- patience가 낮을수록 짧고 예민한 답변을 한다. patience가 매우 낮으면 더 이상 길게 설명하지 않으려 한다.
- 시스템 설명, 상태 수치, JSON, 메타 발언은 절대 출력하지 않는다.
- 이번 답변은 첨부된 조사 정보 지시만을 기준으로 생성한다.
- world context나 별도의 사건 문서를 참조하지 않는다.
- responseDirectivePrompt의 지시를 그대로 따른다.
- responseDirectivePrompt가 회피 지시라면 회피하고, 정보 제공 지시라면 그 범위 안에서만 답한다.
- 첨부된 조사 정보 지시와 최근 대화와 모순되지 않도록 한다.

이제 NPC의 실제 대사만 출력해라.
""".strip()
    return prompt

def _format_summary_context(summary_context: Dict[str, Any] | None) -> str:
    if not summary_context:
        return "- 없음"

    summary = _safe_reply_text(summary_context.get("summary") or "")
    user_goal = _safe_reply_text(summary_context.get("user_goal") or "")
    topics = summary_context.get("topics") or []
    relationship_state = summary_context.get("relationship_state") or {}

    lines = []
    if summary:
        lines.append(f"- summary: {summary}")
    if user_goal:
        lines.append(f"- user_goal: {user_goal}")
    if topics:
        lines.append(f"- topics: {', '.join(str(topic) for topic in topics[:8])}")
    if relationship_state:
        lines.append(f"- relationship_state: {json.dumps(relationship_state, ensure_ascii=False)}")

    return os.linesep.join(lines) if lines else "- 없음"


def _format_world_context(world_context_chunks: List[str]) -> str:
    if not world_context_chunks:
        return "- 없음"

    formatted = []
    for index, chunk in enumerate(world_context_chunks[:4], start=1):
        formatted.append(f"- chunk{index}: {_safe_reply_text(chunk)}")
    return os.linesep.join(formatted)


def _build_investigation_error_response(error_message: str):
    return {
        "ok": False,
        "error": str(error_message),
        "replyText": "",
        "interrogationState": _default_interrogation_state(),
        "stateDelta": {
            "unlockTopicIds": [],
            "markStatements": [],
        },
        "presentationHints": {},
    }


def _build_reply_error_response(error_message: str):
    return {
        "ok": False,
        "turnId": "",
        "error": str(error_message),
        "replyText": "",
        "conversationState": _default_conversation_state(),
        "stateDelta": {
            "unlockTopicIds": [],
            "markStatements": [],
        },
        "presentationHints": {},
    }


def _build_tell_error_response(error_message: str, turn_id: str = ""):
    return {
        "ok": False,
        "turnId": turn_id,
        "error": str(error_message),
        "tellResult": {
            "turnId": turn_id,
            "tell": 0.0,
            "band": "Stable",
            "primaryAction": "unknown",
            "reason": "",
        },
    }


def _load_world_chunks():
    chunks = []
    print(f"[PATH] WORLD_DIR={WORLD_DIR}")
    if not os.path.isdir(WORLD_DIR):
        print(f"[WARN] WORLD_DIR not found: {WORLD_DIR}")
        return chunks

    world_files = []
    for fname in os.listdir(WORLD_DIR):
        path = os.path.join(WORLD_DIR, fname)
        if os.path.isfile(path) and any(fname.lower().endswith(ext) for ext in [".txt", ".md"]):
            world_files.append(path)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    chunks.extend([p for p in text.split("\n\n") if p.strip()])

    print(f"[PATH] WORLD_FILES={len(world_files)}")
    for world_file in world_files:
        print(f"[WORLD] {world_file}")
    return chunks


def init_index():
    print(f"[PATH] BASE_DIR={BASE_DIR}")
    print(f"[PATH] DATA_DIR={DATA_DIR}")
    print(f"[PATH] PDF_PATH={PDF_PATH}")
    print(f"[PATH] INDEX_PATH={INDEX_PATH}")
    print(f"[PATH] WORLD_DIR={WORLD_DIR}")

    if not os.path.exists(INDEX_PATH + ".index"):
        world_chunks = _load_world_chunks()
        if world_chunks:
            print(f"세계관 문서 {len(world_chunks)}개 청크 적재")
            retriever.build_index(world_chunks)
            retriever.save_index(INDEX_PATH)
            print("인덱스 로드 완료")
        else:
            print("세계관 문서 없음")
    else:
        print(f"[PATH] Loading existing index from {INDEX_PATH}.index")
        retriever.load_index(INDEX_PATH)
        print("기존 인덱스 로드 완료")


@app.route("/log/summary", methods=["GET"])
def log_summary():
    user_id = request.args.get("user_id", "anonymous")
    persona_key = request.args.get("personaKey", "")
    hist = _read_scoped_log(user_id, persona_key)
    history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]
    res = summarize_context(history, recent_turns=20, max_tokens=384)
    return jsonify(res)


@app.route("/log/summary-llm", methods=["GET"])
def log_summary_llm():
    user_id = request.args.get("user_id", "anonymous")
    persona_key = request.args.get("personaKey", "")
    hist = _read_scoped_log(user_id, persona_key)
    history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]
    result = summarize_context_llm(
        history=history,
        recent_turns=20,
        model="gemma3:12b",
        max_tokens=384,
        fallback_rule_based=True,
        rule_based_fn=summarize_context_rule_based,
    )
    return jsonify(result)


@app.route("/log/reset", methods=["POST"])
def log_reset():
    data = request.get_json() or {}
    user_id = data.get("user_id", "anonymous")
    persona_key = data.get("personaKey", "")
    _reset_scoped_log(user_id, persona_key)
    return jsonify({"ok": True})


def _investigation_stream_chunk(chunk_type, **payload):
    chunk = {"type": chunk_type}
    chunk.update(payload)
    return json.dumps(chunk, ensure_ascii=False) + "\n"


def _reply_stream_chunk(chunk_type, **payload):
    chunk = {"type": chunk_type}
    chunk.update(payload)
    return json.dumps(chunk, ensure_ascii=False) + "\n"


@app.route("/investigation/npc/reply", methods=["POST"])
def investigation_npc_reply():
    try:
        payload = request.get_json() or {}
        turn_id = payload.get("turnId") or ""
        user_id = payload.get("playerId") or payload.get("userId") or "anonymous"
        npc_id = payload.get("npcId") or "npc"
        persona_key = payload.get("personaKey") or npc_id or ""
        phase = payload.get("phase") or "investigation"
        interaction = payload.get("interaction") or {}
        action_type = interaction.get("actionType") or "Talk"
        player_intent_text = interaction.get("playerIntentText") or ""
        scene_state = payload.get("sceneState") or {}
        npc_local_state = payload.get("npcLocalState") or {}
        conversation_context = payload.get("conversationContext") or {}
        interrogation_profile = payload.get("interrogationProfile") or {}

        if not persona_key:
            return jsonify(_build_reply_error_response("personaKey가 없습니다.")), 400

        persona = load_persona(persona_key)
        if persona is None:
            return jsonify(_build_reply_error_response(f"persona not found: {persona_key}")), 404

        if not player_intent_text and action_type == "Talk":
            return jsonify(_build_reply_error_response("interaction.playerIntentText가 없습니다.")), 400

        print(
            f"[INVESTIGATION][REPLY] phase={phase} user={user_id} npc={npc_id} "
            f"persona={persona_key} action={action_type} turnId={turn_id}"
        )

        _append_scoped_log(
            user_id,
            persona_key,
            "user",
            f"[INVESTIGATION/{npc_id}/{action_type}] {_safe_reply_text(player_intent_text)}",
        )

        attached_item_directive = _normalize_attached_item_directive(interaction)
        use_attached_directive = bool(attached_item_directive.get("responseDirectivePrompt"))

        conversation_state = _derive_conversation_state(
            action_type,
            interaction,
            scene_state,
            npc_local_state,
            conversation_context,
            interrogation_profile,
        )

        if use_attached_directive:
            summary_context = None
            world_context_chunks = []
            print(
                "[INVESTIGATION][REPLY_MODE] "
                f"mode=attached npc={npc_id} itemId={attached_item_directive.get('itemId', '')} "
                f"thresholdSatisfied={attached_item_directive.get('thresholdSatisfied', False)} "
                f"followingItemId={attached_item_directive.get('followingItemId', '')}"
            )
            prompt = _build_reply_prompt_attached(
                payload,
                persona,
                conversation_state,
                attached_item_directive,
            )
        else:
            summary_context = _read_scoped_summary(user_id, persona_key)
            world_context_chunks = retriever.search(player_intent_text) if player_intent_text else []
            print(f"[INVESTIGATION][REPLY_MODE] mode=general npc={npc_id}")
            prompt = _build_reply_prompt_general(
                payload,
                persona,
                conversation_state,
                summary_context,
                world_context_chunks,
            )
        statement_id = _build_statement_id(npc_id, npc_local_state)
        message_id = f"{statement_id}_stream"
        start_time = time.time()

        def _update_summary_background():
            try:
                hist = _read_scoped_log(user_id, persona_key, max_items=60)
                history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]

                summary = summarize_context_llm(
                    history=history,
                    recent_turns=20,
                    model=ACTIVE_MODEL,
                    max_tokens=256,
                    fallback_rule_based=True,
                    rule_based_fn=summarize_context_rule_based,
                )
                _write_scoped_summary(user_id, persona_key, summary)
                print(f"[SUMMARY] updated for user={user_id}, persona={persona_key}")
            except Exception as e:
                print(f"[WARN] summary update failed after reply stream: {e}")

        def generate():
            first_chunk_logged = False
            streamed_text = ""

            yield _reply_stream_chunk(
                "start",
                messageId=message_id,
                npcDisplayName=(persona.get("identity") or {}).get("name", npc_id),
            )

            yield _reply_stream_chunk(
                "conversationState",
                messageId=message_id,
                conversationState=conversation_state,
            )

            try:
                for chunk in _query_active_llm_stream(prompt):
                    if not first_chunk_logged:
                        latency = time.time() - start_time
                        print(
                            f"[LATENCY] reply user={user_id} npc={npc_id} "
                            f"persona={persona_key} took {latency:.2f}s to first token"
                        )
                        first_chunk_logged = True

                    streamed_text += chunk
                    yield _reply_stream_chunk(
                        "delta",
                        messageId=message_id,
                        text=chunk,
                    )

                reply_text = _safe_reply_text(streamed_text)
                unlock_topic_ids = []

                response_payload = {
                    "ok": True,
                    "turnId": turn_id,
                    "replyText": reply_text,
                    "conversationState": conversation_state,
                    "stateDelta": {
                        "unlockTopicIds": unlock_topic_ids,
                        "markStatements": [
                            {
                                "statementId": statement_id,
                                "text": reply_text,
                            }
                        ],
                    },
                    "presentationHints": _conversation_state_to_presentation_hints(conversation_state),
                    "error": "",
                }

                yield _reply_stream_chunk(
                    "complete",
                    messageId=message_id,
                    response=response_payload,
                )

                _append_scoped_log(user_id, persona_key, persona_key, reply_text)

                print(
                    "[INVESTIGATION][SUMMARY] "
                    f"npc={npc_id} used={summary_context is not None} "
                    f"summary={json.dumps(_safe_reply_text((summary_context or {}).get('summary', '')), ensure_ascii=False)} "
                    f"user_goal={json.dumps(_safe_reply_text((summary_context or {}).get('user_goal', '')), ensure_ascii=False)} "
                    f"topics={json.dumps((summary_context or {}).get('topics', []), ensure_ascii=False)}"
                )
                print(
                    "[INVESTIGATION][WORLD_CONTEXT] "
                    f"npc={npc_id} chunks={json.dumps([_safe_reply_text(chunk) for chunk in world_context_chunks[:4]], ensure_ascii=False)}"
                )
                print(
                    f"[INVESTIGATION][REPLY][RESULT] npc={npc_id} "
                    f"interest={conversation_state['affect']['interest']} "
                    f"attitude={conversation_state['affect']['attitude']} "
                    f"patience={conversation_state['patience']} "
                    f"unlocks={unlock_topic_ids}"
                )
                print(
                    "[INVESTIGATION][REPLY][TEXT] "
                    f"npc={npc_id} reply={json.dumps(reply_text, ensure_ascii=False)}"
                )

                threading.Thread(target=_update_summary_background, daemon=True).start()

            except Exception as stream_error:
                import traceback

                print("[ERROR] investigation_npc_reply stream 예외 발생:")
                traceback.print_exc()
                yield _reply_stream_chunk(
                    "error",
                    messageId=message_id,
                    error=str(stream_error),
                )

        return Response(
            stream_with_context(generate()),
            content_type="application/x-ndjson; charset=utf-8",
        )

    except Exception as e:
        import traceback

        print("[ERROR] investigation_npc_reply 예외 발생:")
        traceback.print_exc()
        return jsonify(_build_reply_error_response(str(e))), 500


@app.route("/investigation/npc/tell", methods=["POST"])
def investigation_npc_tell():
    try:
        payload = request.get_json() or {}
        turn_id = payload.get("turnId") or ""
        npc_id = payload.get("npcId") or "npc"
        persona_key = payload.get("personaKey") or npc_id or ""
        question_text = payload.get("questionText") or ""

        if not persona_key:
            return jsonify(_build_tell_error_response("personaKey가 없습니다.", turn_id)), 400

        persona = load_persona(persona_key)
        if persona is None:
            return jsonify(_build_tell_error_response(f"persona not found: {persona_key}", turn_id)), 404

        if not question_text:
            return jsonify(_build_tell_error_response("questionText가 없습니다.", turn_id)), 400

        tell_analysis = _derive_tell_llm(question_text, persona)
        tell_result = {
            "turnId": turn_id,
            "tell": tell_analysis["tell"],
            "band": tell_analysis["band"],
            "primaryAction": tell_analysis["primaryAction"],
            "reason": tell_analysis["reason"],
        }

        print(
            "[INVESTIGATION][TELL] "
            f"npc={npc_id} turnId={turn_id} tell={tell_result['tell']} "
            f"band={tell_result['band']} primaryAction={tell_result['primaryAction']}"
        )

        return jsonify({
            "ok": True,
            "turnId": turn_id,
            "tellResult": tell_result,
            "error": "",
        })

    except Exception as e:
        import traceback

        print("[ERROR] investigation_npc_tell 예외 발생:")
        traceback.print_exc()
        return jsonify(_build_tell_error_response(str(e), "")), 500


@app.route("/investigation/npc", methods=["POST"])
def investigation_npc():
    try:
        payload = request.get_json() or {}
        user_id = payload.get("playerId") or payload.get("userId") or "anonymous"
        npc_id = payload.get("npcId") or "npc"
        persona_key = payload.get("personaKey") or npc_id or ""
        phase = payload.get("phase") or "investigation"
        interaction = payload.get("interaction") or {}
        action_type = interaction.get("actionType") or "Talk"
        player_intent_text = interaction.get("playerIntentText") or ""
        scene_state = payload.get("sceneState") or {}
        npc_local_state = payload.get("npcLocalState") or {}
        conversation_context = payload.get("conversationContext") or {}
        interrogation_profile = payload.get("interrogationProfile") or {}

        if not persona_key:
            return jsonify(_build_investigation_error_response("personaKey가 없습니다.")), 400

        persona = load_persona(persona_key)
        if persona is None:
            return jsonify(_build_investigation_error_response(f"persona not found: {persona_key}")), 404

        if not player_intent_text and action_type == "Talk":
            return jsonify(_build_investigation_error_response("interaction.playerIntentText가 없습니다.")), 400

        print(
            f"[INVESTIGATION] phase={phase} user={user_id} npc={npc_id} "
            f"persona={persona_key} action={action_type}"
        )

        _append_scoped_log(
            user_id,
            persona_key,
            "user",
            f"[INVESTIGATION/{npc_id}/{action_type}] {_safe_reply_text(player_intent_text)}",
        )

        interrogation_state = _derive_interrogation_state(
            action_type,
            interaction,
            scene_state,
            npc_local_state,
            conversation_context,
            interrogation_profile,
            persona,
        )
        prompt = _build_investigation_prompt(payload, persona, interrogation_state)
        statement_id = _build_statement_id(npc_id, npc_local_state)
        message_id = f"{statement_id}_stream"
        start_time = time.time()

        def _update_summary_background():
            try:
                hist = _read_scoped_log(user_id, persona_key, max_items=60)
                history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]

                summary = summarize_context_llm(
                    history=history,
                    recent_turns=20,
        model=ACTIVE_MODEL,
                    max_tokens=256,
                    fallback_rule_based=True,
                    rule_based_fn=summarize_context_rule_based,
                )
                _write_scoped_summary(user_id, persona_key, summary)
                print(f"[SUMMARY] updated for user={user_id}, persona={persona_key}")
            except Exception as e:
                print(f"[WARN] summary update failed after investigation stream: {e}")

        def generate():
            first_chunk_logged = False
            streamed_text = ""

            yield _investigation_stream_chunk(
                "start",
                messageId=message_id,
                npcDisplayName=(persona.get("identity") or {}).get("name", npc_id),
            )

            yield _investigation_stream_chunk(
                "state",
                messageId=message_id,
                interrogationState=interrogation_state,
            )

            try:
                for chunk in _query_active_llm_stream(prompt, "gemma3:12b"):
                    if not first_chunk_logged:
                        latency = time.time() - start_time
                        print(
                            f"[LATENCY] investigation user={user_id} npc={npc_id} "
                            f"persona={persona_key} took {latency:.2f}s to first token"
                        )
                        first_chunk_logged = True

                    streamed_text += chunk
                    yield _investigation_stream_chunk(
                        "delta",
                        messageId=message_id,
                        text=chunk,
                    )

                reply_text = _safe_reply_text(streamed_text)
                unlock_topic_ids = []

                response_payload = {
                    "ok": True,
                    "replyText": reply_text,
                    "interrogationState": interrogation_state,
                    "stateDelta": {
                        "unlockTopicIds": unlock_topic_ids,
                        "markStatements": [
                            {
                                "statementId": statement_id,
                                "text": reply_text,
                            }
                        ],
                    },
                    "presentationHints": _interrogation_state_to_presentation_hints(interrogation_state),
                    "error": "",
                }

                yield _investigation_stream_chunk(
                    "complete",
                    messageId=message_id,
                    response=response_payload,
                )

                _append_scoped_log(user_id, persona_key, persona_key, reply_text)

                print(
                    "[INVESTIGATION][BIOMETRIC] "
                    f"npc={npc_id} state={json.dumps(interrogation_state, ensure_ascii=False)}"
                )
                print(
                    "[INVESTIGATION][TELL] "
                    f"npc={npc_id} band={interrogation_state.get('tellMeta', {}).get('band', 'Stable')} "
                    f"primaryAction={interrogation_state.get('tellMeta', {}).get('primaryAction', 'unknown')} "
                    f"reason={json.dumps(interrogation_state.get('tellMeta', {}).get('reason', ''), ensure_ascii=False)}"
                )
                print(
                    "[INVESTIGATION][TEXT] "
                    f"npc={npc_id} reply={json.dumps(reply_text, ensure_ascii=False)}"
                )
                print(
                    f"[INVESTIGATION][RESULT] npc={npc_id} tell={interrogation_state['tell']} "
                    f"interest={interrogation_state['affect']['interest']} "
                    f"attitude={interrogation_state['affect']['attitude']} "
                    f"patience={interrogation_state['patience']} "
                    f"unlocks={unlock_topic_ids}"
                )

                threading.Thread(target=_update_summary_background, daemon=True).start()

            except Exception as stream_error:
                import traceback

                print("[ERROR] investigation_npc stream 예외 발생:")
                traceback.print_exc()
                yield _investigation_stream_chunk(
                    "error",
                    messageId=message_id,
                    error=str(stream_error),
                )

        return Response(
            stream_with_context(generate()),
            content_type="application/x-ndjson; charset=utf-8",
        )

    except Exception as e:
        import traceback

        print("[ERROR] investigation_npc 예외 발생:")
        traceback.print_exc()
        return jsonify(_build_investigation_error_response(str(e))), 500


@app.route("/ask-stream", methods=["POST"])
def ask_stream():
    try:
        data = request.get_json() or {}
        question = data.get("question")
        user_id = data.get("user_id", "anonymous")
        persona_key = data.get("personaKey") or ""
        print(f"[DEBUG] 질문: {question}, 유저: {user_id}, 페르소나: {persona_key or 'assistant'}")

        if not question:
            return "질문이 없습니다", 400

        start_time = time.time()
        _append_scoped_log(user_id, persona_key, "user", question)

        top_chunks = retriever.search(question)
        persona = load_persona(persona_key) if persona_key else None

        summary = _read_scoped_summary(user_id, persona_key)
        merged_chunks = top_chunks
        if summary and (summary.get("summary") or "").strip():
            summary_line = f"[대화요약] {summary['summary']}"
            merged_chunks = [summary_line] + top_chunks

        prompt = build_prompt_v2(merged_chunks, question, persona)

        for i, c in enumerate(top_chunks, 1):
            head = (c[:20] + "…") if len(c) > 20 else c
            print(f"[{i}] {head}")
        print("=================================\n")

        def _update_summary_background():
            try:
                hist = _read_scoped_log(user_id, persona_key, max_items=60)
                history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]

                summary = summarize_context_llm(
                    history=history,
                    recent_turns=20,
                    model=ACTIVE_MODEL,
                    max_tokens=256,
                    fallback_rule_based=True,
                    rule_based_fn=summarize_context_rule_based,
                )
                _write_scoped_summary(user_id, persona_key, summary)
                print(f"[SUMMARY] updated for user={user_id}, persona={persona_key}")
            except Exception as e:
                print(f"[WARN] summary update failed: {e}")

        def generate():
            nonlocal start_time
            first_chunk_logged = False
            buffer = []
            for chunk in _query_active_llm_stream(prompt):
                if not first_chunk_logged:
                    latency = time.time() - start_time
                    print(f"[LATENCY] user={user_id} persona={persona_key or 'assistant'} took {latency:.2f}s to first token")
                    first_chunk_logged = True
                buffer.append(chunk)
                yield chunk + "\n"

            full_answer = "".join(buffer).strip()
            speaker = persona_key if persona_key else "assistant"
            try:
                _append_scoped_log(user_id, persona_key, speaker, full_answer)
            except Exception as log_err:
                print(f"[WARN] 답변 로그 기록 실패: {log_err}")

            threading.Thread(target=_update_summary_background, daemon=True).start()

        return Response(stream_with_context(generate()), content_type="text/plain")

    except Exception as e:
        import traceback

        print("[ERROR] ask_stream 예외 발생:")
        traceback.print_exc()
        return f"[SERVER ERROR] {str(e)}", 500


if __name__ == "__main__":
    from waitress import serve

    _select_startup_model()
    init_index()
    serve(app, host="0.0.0.0", port=5000)
    # app.run(port=5000, debug=True)
