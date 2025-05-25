import requests

def build_prompt(chunks: list, query: str) -> str:
    """검색된 문단들과 질문을 하나의 프롬프트로 구성"""
    context = "\n\n".join(chunks)
    prompt = (
        "다음 문서를 참고하여 질문에 답변하세요.\n\n"
        f"문서:\n{context}\n\n"
        f"질문: {query}\n"
        "답변:"
    )
    return prompt

def query_ollama(prompt: str, model: str = "mistral", host: str = "http://localhost:11434") -> str:
    """Ollama에 프롬프트를 보내고 응답을 받아옴"""
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as e:
        return f"[ERROR] Ollama 호출 실패: {e}"
