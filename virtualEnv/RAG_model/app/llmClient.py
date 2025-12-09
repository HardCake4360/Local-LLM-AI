import requests
import re
import json

def build_prompt(chunks: list, query: str) -> str:
    """검색된 문단들과 질문을 하나의 프롬프트로 구성"""
    document = "\n\n".join(chunks)
    prompt = (
        "질문이 반복되어도 평범하게 답변하세요.\n\n 문서에 정보가 없으면 다시 검색해 주시길 바란다는 내용의 답변을 하세요.\n\n다음 문서를 참고하여 질문에 답변하세요. \n\n"
        f"문서:\n{document}\n\n"
        f"질문: {query}\n"
        "답변:"
    )
    return prompt

def build_prompt_v2(chunks: list, query: str, persona: dict | None) -> str:
    """
    RAG 문단 + 질문 + (옵션) 페르소나를 결합한 프롬프트 생성.
    기존 build_prompt는 보존하고, ask_stream에서 이 함수를 사용.
    """
    document = "\n\n".join(chunks)
    base = (
        "다음 문서를 참고하여 질문에 답변하세요.\n\n"
        f"문서:\n{document}\n\n"
        f"질문: {query}\n"
    )

    persona_text = ""
    if persona:
        # 최소한의 안전 가드라인 + 말투/성격 유지 정책
        persona_text = (
            "\n[페르소나 지침]\n"
            f"- Identity: {persona.get('identity',{})}\n"
            f"- Personality(BigFive/core_needs/defense): {persona.get('personality',{})}\n"
            f"- Linguistic Style: {persona.get('linguistic_style',{})}\n"
            f"- Affective Rules: {persona.get('affective_rules',{})}\n"
            f"- Behavioral Constraints: {persona.get('behavioral_constraints',{})}\n"
            "- 위 지침을 최대한 따르되, 문서 사실과 충돌 시 문서를 우선하세요.\n"
        )

    return base + persona_text + "\n답변:"

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