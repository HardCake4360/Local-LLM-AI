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
PDF_PATH = "app/data/pdfs/" + PDF_NAME + ".pdf"  # PDF 경로
INDEX_PATH = "app/data/index/world"
WORLD_DIR = "app/data/world"

DEFAULT_SIGNAL = {
    "stress": 0.0,
    "distortion": 0.0,
    "focus": 0.0,
}

retriever = Retriever()


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "..."


def _signal_to_presentation_hints(signal: Dict[str, float]) -> Dict[str, Any]:
    stress = _clamp(_to_float(signal.get("stress"), 0.0))
    distortion = _clamp(_to_float(signal.get("distortion"), 0.0))
    focus = _clamp(_to_float(signal.get("focus"), 0.0))

    if stress >= 0.7 or distortion >= 0.6:
        animation = "tense_idle"
    elif stress >= 0.4:
        animation = "guarded_idle"
    else:
        animation = "calm_idle"

    if focus >= 0.75 and stress <= 0.35:
        voice_tone = "calm"
    elif stress >= 0.65:
        voice_tone = "strained"
    else:
        voice_tone = "guarded"

    return {
        "animation": animation,
        "voiceTone": voice_tone,
        "uiNoiseLevel": round(distortion, 2),
    }


def _derive_signal(action_type: str, last_signal: Dict[str, Any], reply_text: str) -> Dict[str, float]:
    base = {
        "stress": _clamp(_to_float((last_signal or {}).get("stress"), 0.2)),
        "distortion": _clamp(_to_float((last_signal or {}).get("distortion"), 0.1)),
        "focus": _clamp(_to_float((last_signal or {}).get("focus"), 0.7)),
    }

    action = (action_type or "Talk").strip()
    if action == "PresentEvidence":
        base["stress"] += 0.18
        base["distortion"] += 0.14
        base["focus"] -= 0.06
    elif action == "AskTopic":
        base["stress"] += 0.10
        base["distortion"] += 0.07
        base["focus"] += 0.01
    else:  # Talk
        base["stress"] += 0.03
        base["distortion"] += 0.02
        base["focus"] += 0.01

    text = reply_text or ""
    if re.search(r"노이즈|잡음|화이트\s*노이즈|불확실|흐릿|애매", text):
        base["stress"] += 0.06
        base["distortion"] += 0.05

    if re.search(r"보청기", text):
        base["stress"] += 0.08
        base["distortion"] += 0.04

    if re.search(r"CRT|브라운관", text, re.IGNORECASE):
        base["stress"] += 0.07
        base["distortion"] += 0.06

    if re.search(r"분명|확실|봤다|기억한다|기억해|똑똑히", text):
        base["focus"] += 0.08
        base["distortion"] -= 0.03

    if re.search(r"모른다|잘 모르겠|장담 못|확신은 없", text):
        base["focus"] -= 0.05
        base["distortion"] += 0.04

    return {
        "stress": round(_clamp(base["stress"]), 2),
        "distortion": round(_clamp(base["distortion"]), 2),
        "focus": round(_clamp(base["focus"]), 2),
    }


def _extract_unlock_topics(interaction: Dict[str, Any], reply_text: str) -> List[str]:
    unlocks: List[str] = []
    text = reply_text or ""
    action_type = (interaction or {}).get("actionType") or "Talk"
    evidence_id = (interaction or {}).get("evidenceId")

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


def _build_statement_id(npc_id: str, npc_state: Dict[str, Any]) -> str:
    count = int(_to_float((npc_state or {}).get("conversationCount"), 0)) + 1
    safe_npc = re.sub(r"[^a-zA-Z0-9_]+", "_", npc_id or "npc").strip("_") or "npc"
    return f"{safe_npc}_stmt_{count:03d}"


def _build_investigation_prompt(payload: Dict[str, Any], persona: Dict[str, Any]) -> str:
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

    known_clues = scene_state.get("knownClues") or []
    unlocked_topics = scene_state.get("unlockedTopics") or []
    last_signal = (npc_local_state.get("lastKnownSignal") or DEFAULT_SIGNAL)

    persona_json = json.dumps(persona or {}, ensure_ascii=False, indent=2)
    scene_state_json = json.dumps(scene_state, ensure_ascii=False, indent=2)
    npc_state_json = json.dumps(npc_local_state, ensure_ascii=False, indent=2)

    prompt = f"""
너는 한국어 추리 게임 속 NPC다. 지금부터 조사 파트 전용 응답만 생성한다.

[캐릭터 / 페르소나]
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

[이전 signal]
- stress: {last_signal.get('stress', 0.0)}
- distortion: {last_signal.get('distortion', 0.0)}
- focus: {last_signal.get('focus', 0.0)}

[최근 대화]
{os.linesep.join(recent_lines) if recent_lines else '- 없음'}

[응답 원칙]
- 플레이어가 증거를 제시하면 당황, 경계, 회피, 혹은 제한적 인정 중 하나를 보여줄 수 있다.
- knownClues, unlockedTopics와 모순되지 않도록 한다.

이제 NPC의 실제 대사만 출력해라.
""".strip()
    return prompt


def _build_investigation_error_response(error_message: str):
    return {
        "ok": False,
        "error": str(error_message),
        "replyText": "",
        "signal": dict(DEFAULT_SIGNAL),
        "stateDelta": {
            "unlockTopicIds": [],
            "markStatements": [],
        },
        "presentationHints": {},
    }


def _load_world_chunks():
    chunks = []
    if not os.path.isdir(WORLD_DIR):
        return chunks
    for fname in os.listdir(WORLD_DIR):
        path = os.path.join(WORLD_DIR, fname)
        if os.path.isfile(path) and any(fname.lower().endswith(ext) for ext in [".txt", ".md"]):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    # 간단히 라인 단위 나누기
                    chunks.extend([p for p in text.split("\n\n") if p.strip()])
    return chunks


def init_index():
    # 세계관 문서 인덱싱
    if not os.path.exists(INDEX_PATH + ".index"):
        world_chunks = _load_world_chunks()
        if world_chunks:
            print(f" 세계관 문서 {len(world_chunks)}개 청크 적재")
            retriever.build_index(world_chunks)
            retriever.save_index(INDEX_PATH)
            print("인덱스 로드 완료")
        else:
            print("세계관 문서 없음")
    else:
        retriever.load_index(INDEX_PATH)
        print("📚 기존 인덱스 로드 완료")


@app.route("/log/summary", methods=["GET"])
def log_summary():
    user_id = request.args.get("user_id", "anonymous")
    hist = read_log(user_id)
    # speaker/text 필드 이름 맞춰 요약
    history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]
    res = summarize_context(history, recent_turns=20, max_tokens=384)
    return jsonify(res)


@app.route("/log/summary-llm", methods=["GET"])
def log_summary_llm():
    user_id = request.args.get("user_id", "anonymous")
    hist = read_log(user_id)
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
    reset_log(user_id)
    return jsonify({"ok": True})



def _investigation_stream_chunk(chunk_type, **payload):
    chunk = {"type": chunk_type}
    chunk.update(payload)
    return json.dumps(chunk, ensure_ascii=False) + "\n"

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
        npc_local_state = payload.get("npcLocalState") or {}
        last_signal = npc_local_state.get("lastKnownSignal") or DEFAULT_SIGNAL

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

        prompt = _build_investigation_prompt(payload, persona)
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
            raw_chunks = []
            streamed_text = ""

            yield _investigation_stream_chunk(
                "start",
                messageId=message_id,
                npcDisplayName=(persona.get("identity") or {}).get("name", npc_id),
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

                    raw_chunks.append(chunk)
                    streamed_text += chunk
                    yield _investigation_stream_chunk(
                        "delta",
                        messageId=message_id,
                        text=chunk,
                    )

                reply_text = _safe_reply_text(streamed_text)
                signal = _derive_signal(action_type, last_signal, reply_text)
                unlock_topic_ids = _extract_unlock_topics(interaction, reply_text)

                response_payload = {
                    "ok": True,
                    "replyText": reply_text,
                    "signal": signal,
                    "stateDelta": {
                        "unlockTopicIds": unlock_topic_ids,
                        "markStatements": [
                            {
                                "statementId": statement_id,
                                "text": reply_text,
                            }
                        ],
                    },
                    "presentationHints": _signal_to_presentation_hints(signal),
                    "error": "",
                }

                yield _investigation_stream_chunk(
                    "signal",
                    messageId=message_id,
                    signal=signal,
                )

                yield _investigation_stream_chunk(
                    "complete",
                    messageId=message_id,
                    response=response_payload,
                )

                append_log(user_id, persona_key, reply_text)

                print(
                    f"[INVESTIGATION][RESULT] npc={npc_id} stress={signal['stress']} "
                    f"distortion={signal['distortion']} focus={signal['focus']} "
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

        # 요청 시각 기록
        start_time = time.time()

        # 로그 기록: 유저 발화
        append_log(user_id, "user", question)

        # RAG 검색
        top_chunks = retriever.search(question)

        # 페르소나 로드(없으면 None)
        persona = load_persona(persona_key) if persona_key else None

        # 최근 로그 요약을 질문 보조 컨텍스트로 사용
        hist = read_log(user_id, max_items=60)
        history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]

        # 캐시 사용 방식: 저장된 요약 캐시를 사용 (없으면 생략)
        summary = read_summary(user_id)
        merged_chunks = top_chunks
        if summary and (summary.get("summary") or "").strip():
            summary_line = f"[대화요약] {summary['summary']}"
            merged_chunks = [summary_line] + top_chunks

        # 페르소나 포함 프롬프트
        prompt = build_prompt_v2(merged_chunks, question, persona)

        for i, c in enumerate(top_chunks, 1):
            head = (c[:20] + "…") if len(c) > 20 else c
            print(f"[{i}] {head}")
        print("=================================\n")

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

                summary_text = summary.get("summary") or ""
                rel = summary.get("relationship_state", {})
                mode = "LLM" if summary.get("user_goal") is not None else "RULE-BASED(FALLBACK)"

                print("\n[SUMMARY UPDATE] =======================")
                print(f"User: {user_id} | Persona: {persona_key or 'assistant'}")
                print(f"[MODE] {mode}")
                print(f"[요약] {summary_text}")
                print(f"[USER_GOAL] {summary.get('user_goal')}")
                print(f"[대화 주제] {', '.join(summary.get('topics', []))}")
                print(f"[관계 상태] {rel}")
                print("=======================================\n")
            except Exception as e:
                print(f"[WARN] summary update failed: {e}")

        def generate():
            nonlocal start_time
            first_chunk_logged = False
            buffer = []
            # 모델 이름은 기존과 동일하게 사용, 필요 시 config로 분리
            for chunk in query_ollama_stream(prompt, "gemma3:12b"):
                if not first_chunk_logged:
                    latency = time.time() - start_time
                    # 지연시간 로그 출력 또는 저장
                    print(f"[LATENCY] user={user_id} persona={persona_key or 'assistant'} took {latency:.2f}s to first token")
                    first_chunk_logged = True
                buffer.append(chunk)
                yield chunk + "\n"

            # 스트리밍 종료 후 한번에 답변 기록
            full_answer = "".join(buffer).strip()
            speaker = persona_key if persona_key else "assistant"
            try:
                append_log(user_id, speaker, full_answer)
            except Exception as log_err:
                print(f"[WARN] 답변 로그 기록 실패: {log_err}")

            # 스트리밍 완료 후 컨택스트 요약 갱신
            threading.Thread(target=_update_summary_background, daemon=True).start()

        return Response(stream_with_context(generate()), content_type="text/plain")

    except Exception as e:
        import traceback
        print("[ERROR] ask_stream 예외 발생:")
        traceback.print_exc()
        return f"[SERVER ERROR] {str(e)}", 500


if __name__ == "__main__":
    from waitress import serve

    init_index()
    serve(app, host="0.0.0.0", port=5000)
    # app.run(port=5000, debug=True)
