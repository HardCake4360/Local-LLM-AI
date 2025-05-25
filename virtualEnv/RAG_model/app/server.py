#요구사항
"""
pip install flask flask-cors
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from documentHandler import extract_text_from_pdf, split_text_to_chunks
from retriever import Retriever
from llmClient import build_prompt, query_ollama

app = Flask(__name__)
CORS(app)

PDF_PATH = "virtualEnv/RAG_model/app/data/test.pdf"   # 미리 준비된 PDF 경로
INDEX_PATH = "virtualEnv/RAG_model/app/data/my_index"

retriever = Retriever()

def init_index():
    """서버 시작 시 PDF 로딩 및 인덱싱"""
    if not os.path.exists(INDEX_PATH + ".index"):
        print("🔍 PDF 인덱스 생성 중...")
        text = extract_text_from_pdf(PDF_PATH)
        chunks = split_text_to_chunks(text)
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

if __name__ == "__main__":
    init_index()
    app.run(port=5000, debug=True)
