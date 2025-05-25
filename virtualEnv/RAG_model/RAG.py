# 요구사항
r"""
패키지 설치
pip install transformers datasets faiss-cpu sentence-transformers requests
C:\Users\user\Documents\GitHub\Local-LLM-AI\virtualEnv\RAG_model
"""

from sentence_transformers import SentenceTransformer, util
import torch
import requests

# 1. 간단한 문서 데이터
documents = [
    "인공지능은 머신러닝과 딥러닝으로 구성됩니다.",
    "파이썬은 인기 있는 프로그래밍 언어입니다.",
    "RAG는 검색을 통해 외부 정보를 사용하는 언어 생성 방식입니다.",
    "키보드의 WASD키 또는 조이스틱을 이용해 이동할 수 있습니다."
]

# 2. 임베딩 모델 로딩
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

# 3. 질문 입력
query = "이동하기 위해선 어떻게 해야 하나요?"
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

# 4. 유사도 기반 검색
cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
top_result_idx = torch.argmax(cos_scores).item()
retrieved_doc = documents[top_result_idx]

# 5. Ollama 연동 생성 함수 정의
def generate_with_ollama(prompt: str, model: str = "mistral") -> str:
    """
    Ollama 서버로 프롬프트를 전송하여 생성된 텍스트를 반환합니다.
    서버는 http://localhost:11434/api/generate 에서 동작해야 합니다.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json().get('response', '').strip()

# 6. 프롬프트 구성 및 생성 실행
prompt = (
    "다음 문서를 읽고 질문에 답하세요.\n"
    f"문서: {retrieved_doc}\n"
    f"질문: {query}\n"
    "답변:"
)
output = generate_with_ollama(prompt)

# 7. 출력
print("🔍 검색된 문서:", retrieved_doc)
print("🧠 생성된 답변:", output)
