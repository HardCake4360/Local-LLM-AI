#요구사항
"""
pip install flask flask-cors
virtualEnv\RAG_model\app\server.py
"""
import time
from persona_store import load_persona, list_persona_keys

from chatlog import append_log, read_log, reset_log, read_summary, write_summary
import threading


from summarizer import summarize_context
from summarizer_llm import summarize_context_llm
from summarizer import summarize_context as summarize_context_rule_based

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
from documentHandler import extract_text_from_pdf, split_text_to_chunks
from retriever import Retriever
from llmClient import build_prompt,build_prompt_v2, query_ollama, query_ollama_stream

app = Flask(__name__)
CORS(app)

PDF_NAME = "DatabasePrompt" #PDF파일 이름
PDF_PATH = "virtualEnv/RAG_model/app/data/pdfs/"+ PDF_NAME +".pdf"   #PDF 경로
INDEX_PATH = "virtualEnv/RAG_model/app/data/index/world"

WORLD_DIR = "virtualEnv/RAG_model/app/data/world"

retriever = Retriever()

def _load_world_chunks():
    chunks = []
    if not os.path.isdir(WORLD_DIR):
        return chunks
    for fname in os.listdir(WORLD_DIR):
        path = os.path.join(WORLD_DIR, fname)
        if os.path.isfile(path) and any(fname.lower().endswith(ext) for ext in [".txt",".md"]):
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
    user_id = request.args.get("user_id","anonymous")
    hist = read_log(user_id)
    # speaker/text 필드 이름 맞춰 요약
    history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]
    res = summarize_context(history, recent_turns=20, max_tokens=384)
    return jsonify(res)

@app.route("/log/summary-llm", methods=["GET"])
def log_summary_llm():
    user_id = request.args.get("user_id","anonymous")
    hist = read_log(user_id)
    history = [{"speaker": h.get("speaker"), "text": h.get("text")} for h in hist]
    result = summarize_context_llm(
        history=history,
        recent_turns=20,
        model="gemma3:12b",
        max_tokens=384,
        fallback_rule_based=True,
        rule_based_fn=summarize_context_rule_based
    )
    return jsonify(result)

@app.route("/log/reset", methods=["POST"])
def log_reset():
    data = request.get_json() or {}
    user_id = data.get("user_id","anonymous")
    reset_log(user_id)
    return jsonify({"ok": True})

@app.route("/ask-stream", methods=["POST"])
def ask_stream():
    try:
        data = request.get_json() or {}
        question   = data.get("question")
        user_id    = data.get("user_id", "anonymous")
        persona_key= data.get("personaKey") or ""
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
        
        #rule based
        #summary = summarize_context(history, recent_turns=20, max_tokens=256)
        
        # #llm based
        # summary = summarize_context_llm(
        #     history=history,
        #     recent_turns=20,
        #     model="gemma3:12b",
        #     max_tokens=256,
        #     fallback_rule_based=True,
        #     rule_based_fn=summarize_context_rule_based
        # )
        # summary_line = f"[대화요약] {summary['summary']}"
        # merged_chunks = [summary_line] + top_chunks

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
            
            #스트리밍 종료 후 한번에 답변 기록
            full_answer = "".join(buffer).strip()
            speaker = persona_key if persona_key else "assistant"
            try:
                append_log(user_id, speaker, full_answer)
            except Exception as log_err:
                print(f"[WARN] 답변 로그 기록 실패: {log_err}")
            
            #스트리밍 완료 후 컨택스트 요약 갱신
            threading.Thread(target=_update_summary_background, daemon=True).start()

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
                    rule_based_fn=summarize_context_rule_based
                )
                write_summary(user_id, summary)
                print(f"[SUMMARY] updated for user={user_id}")
                
                summary_text = summary.get("summary") or ""
                rel = summary.get("relationship_state", {})
                trust = rel.get("user_trust", 0.0)
                closeness = rel.get("closeness", "unknown")

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

        # ★ 로그 기록: 시스템이 보낸 프롬프트 일부를 저장하진 않고,
        #   스트리밍 완료 후 모델 응답을 합쳐 저장하려면 클라이언트에서 수신 후 /log/append 호출 권장
        return Response(stream_with_context(generate()), content_type='text/plain')

    except Exception as e:
        import traceback
        print("[ERROR] ask_stream 예외 발생:")
        traceback.print_exc()
        return f"[SERVER ERROR] {str(e)}", 500

if __name__ == "__main__":
    from waitress import serve
    init_index()
    serve(app,host="0.0.0.0", port=5000)
    #app.run(port=5000, debug=True)
