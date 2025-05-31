import requests
import re
import json

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
        raw_output = response.json().get("response", "").strip()
        return addEndMarkers(raw_output)  # <-- 후처리 적용
    except requests.RequestException as e:
        return f"[ERROR] Ollama 호출 실패: {e}"
    
def query_ollama_stream(prompt: str, model: str = "mistral", host: str = "http://localhost:11434"):
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    try:
        with requests.post(url, json = payload, stream = True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = line.decode("utf-8")
                        data = json.loads(chunk)
                        yield data.get("response", "")
                    except:
                        continue
    except requests.RequestException as e:
        yield f"[ERROR] Ollama 호출 실패: {e}"

def addEndMarkers(text: str) -> str:
    text = re.sub(r'([.!?])\s+', r'\1<END> ', text.strip())
    if not text.endswith(" <END>"):
        text += " <END>"
    return text