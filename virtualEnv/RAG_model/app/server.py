#요구사항
"""
pip install flask flask-cors
virtualEnv\RAG_model\app\server.py
"""
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
from documentHandler import extract_text_from_pdf, split_text_to_chunks, split_by_paragraph
from retriever import Retriever
from llmClient import build_prompt, query_ollama, query_ollama_stream

app = Flask(__name__)
CORS(app)

PDF_NAME = "TetoPrompt" #PDF파일 이름
PDF_PATH = "virtualEnv/RAG_model/app/data/"+ PDF_NAME +".pdf"   #PDF 경로
INDEX_PATH = "virtualEnv/RAG_model/app/data/my_index"

retriever = Retriever()

def init_index():
    """서버 시작 시 PDF 로딩 및 인덱싱"""
    if not os.path.exists(INDEX_PATH + ".index"):
        print("🔍 PDF 인덱스 생성 중...")
        text = extract_text_from_pdf(PDF_PATH)
        print(f"[DEBUG] PDF 텍스트 길이: {len(text)}")
        chunks = split_by_paragraph(text)
        print(f"[DEBUG] 생성된 chunk 수: {len(chunks)}")
        for i, c in enumerate(chunks[:5]):
            print(f"[Chunk {i}] {c[:20]}...")
        retriever.build_index(chunks)
        retriever.save_index(INDEX_PATH)
        print("✅ 인덱스 생성 완료")
    else:
        retriever.load_index(INDEX_PATH)
        print("📚 기존 인덱스 로딩 완료")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "질문이 없습니다."}), 400

    try:
        top_chunks = retriever.search(question)
        prompt = build_prompt(top_chunks, question)
        answer = query_ollama(prompt)
        return jsonify({"answer": answer, "context": top_chunks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask-stream", methods=["POST"])
def ask_stream():
    data = request.get_json()
    print("수신한 JSON 데이터:", data)
    question = data.get("question")
    if not question:
        return "질문이 없습니다",400
    
    top_chunks = retriever.search(question)
    print("[DEBUG] 검색된 chunk:")
    for c in top_chunks:
        print(f"{c[:100]}...")

    prompt = build_prompt(top_chunks, question)
    
    def generate():
        for chunk in query_ollama_stream(prompt,"gemma3:12b"):
            print("서버가 전송 중:", chunk)
            yield chunk + "\n" #줄단위 전송
    
    return Response(stream_with_context(generate()),content_type='text/plain')


if __name__ == "__main__":
    from waitress import serve
    init_index()
    serve(app,host="0.0.0.0", port=5000)
    #app.run(port=5000, debug=True)
