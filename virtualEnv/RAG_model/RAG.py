#요구사항
"""
패키지 설치
pip install transformers datasets faiss-cpu sentence-transformers

"""

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# 1. 간단한 문서 데이터
documents = [
    "서울은 대한민국의 수도입니다.",
    "인공지능은 머신러닝과 딥러닝으로 구성됩니다.",
    "파이썬은 인기 있는 프로그래밍 언어입니다.",
    "RAG는 검색을 통해 외부 정보를 사용하는 언어 생성 방식입니다."
]

# 2. 임베딩 모델 로딩
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

# 3. 질문 입력
query = "RAG는 무엇인가요?"
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

# 4. 유사도 기반 검색
cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
top_result_idx = torch.argmax(cos_scores).item()
retrieved_doc = documents[top_result_idx]

# 5. LLM (간단한 파이프라인 사용)
generator = pipeline("text-generation", model="gpt2")
prompt = f"문서: {retrieved_doc}\n질문: {query}\n답변:"
output = generator(prompt, max_new_tokens=50, do_sample=False)[0]['generated_text']

# 6. 출력
print("🔍 검색된 문서:", retrieved_doc)
print("🧠 생성된 답변:", output)
