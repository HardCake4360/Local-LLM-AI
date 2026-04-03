#요구사항
"""
pip install flask flask-cors
virtualEnv\RAG_model\app\server.py
"""
import json
import os
import re
import threading
import time
from typing import Any, Dict, List

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from persona_store import load_persona, list_persona_keys
from chatlog import append_log, read_log, reset_log, read_summary, write_summary
from summarizer import summarize_context
from summarizer_llm import summarize_context_llm
from summarizer import summarize_context as summarize_context_rule_based
from documentHandler import extract_text_from_pdf, split_text_to_chunks
from retriever import Retriever
from llmClient import build_prompt, build_prompt_v2, query_ollama, query_ollama_stream

app = Flask(__name__)
CORS(app)

PDF_NAME = "DatabasePrompt"  # PDF파일 이름
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "pdfs", PDF_NAME + ".pdf")
INDEX_DIR = os.path.join(DATA_DIR, "index")
INDEX_PATH = os.path.join(INDEX_DIR, "world")
WORLD_DIR = os.path.join(DATA_DIR, "world")

retriever = Retriever()
os.makedirs(INDEX_DIR, exist_ok=True)


def _default_interrogation_state() -> Dict[str, Any]:
    return {
        "tell": 0.0,
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
            state["tell"] += _to_float(rule.get("tellDelta"), 0.0)
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
            state["tell"] += _to_float(rule.get("knownTellDelta"), 0.0)
            state["affect"]["interest"] += _to_float(rule.get("knownInterestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("knownAttitudeDelta"), 0.0)
        else:
            state["tell"] += _to_float(rule.get("unknownTellDelta"), 0.0)
            state["affect"]["interest"] += _to_float(rule.get("unknownInterestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("unknownAttitudeDelta"), 0.0)


def _apply_evidence_rules(evidence_id: str, discovered_evidence_ids: set, profile: Dict[str, Any], state: Dict[str, Any]) -> None:
    if not evidence_id:
        return

    for rule in profile.get("evidenceRules", []):
        if (rule.get("evidenceId") or "") != evidence_id:
            continue

        if evidence_id in discovered_evidence_ids:
            state["tell"] += _to_float(rule.get("discoveredTellDelta"), 0.0)
            state["affect"]["interest"] += _to_float(rule.get("discoveredInterestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("discoveredAttitudeDelta"), 0.0)
        else:
            state["tell"] += _to_float(rule.get("undiscoveredTellDelta"), 0.0)
            state["affect"]["interest"] += _to_float(rule.get("undiscoveredInterestDelta"), 0.0)
            state["affect"]["attitude"] += _to_float(rule.get("undiscoveredAttitudeDelta"), 0.0)


def _derive_interrogation_state(
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
        "tell": 0.0,
        "affect": {
            "interest": last_known_affect["interest"] if npc_local_state else profile.get("baseInterest", 0.0),
            "attitude": last_known_affect["attitude"] if npc_local_state else profile.get("baseAttitude", 0.0),
        },
        "patience": last_known_patience,
    }

    action_weight = _find_action_weight(profile, action_type)
    state["tell"] += _to_float(action_weight.get("tellDelta"), 0.0)
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
        state["tell"] += 0.04
        patience_cost += 2

    state["patience"] = max(0, min(100, state["patience"] - max(1, patience_cost)))
    state["tell"] = round(_clamp(state["tell"], 0.0, 1.0), 2)
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

        append_log(
            user_id,
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
        )
        prompt = _build_investigation_prompt(payload, persona, interrogation_state)
        statement_id = _build_statement_id(npc_id, npc_local_state)
        message_id = f"{statement_id}_stream"
        start_time = time.time()

        def _update_summary_background():
            try:
                hist = read_log(user_id, max_items=60)
                history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]

                summary = summarize_context_llm(
                    history=history,
                    recent_turns=20,
                    model="gemma3:12b",
                    max_tokens=256,
                    fallback_rule_based=True,
                    rule_based_fn=summarize_context_rule_based,
                )
                write_summary(user_id, summary)
                print(f"[SUMMARY] updated for user={user_id}")
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
                for chunk in query_ollama_stream(prompt, "gemma3:12b"):
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
                unlock_topic_ids = _extract_unlock_topics(interaction, reply_text)

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

                append_log(user_id, persona_key, reply_text)

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


if __name__ == "__main__":
    from waitress import serve

    init_index()
    serve(app, host="0.0.0.0", port=5000)
    # app.run(port=5000, debug=True)
