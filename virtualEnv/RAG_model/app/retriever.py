#요구사항
"""
pip install sentence-transformers faiss-cpu
"""
# server/retriever.py
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import os
import pickle

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.text_chunks = []
        self.index = None

    def build_index(self, text_chunks: list):
        """문단을 벡터화하고 FAISS 인덱스 생성"""
        self.text_chunks = text_chunks
        embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        """정적 인덱스 + 유저 인덱스를 모두 검색하여 유사 chunk 반환"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        results = []

        # 검색 결과 (정적)
        if self.index:
            D_static, I_static = self.index.search(query_embedding, top_k)
            static_chunks = [(self.text_chunks[i], D_static[0][j]) for j, i in enumerate(I_static[0])]
            results.extend(static_chunks)

        # 유사도 거리 오름차순 정렬 (L2 거리 기준 → 작을수록 유사함)
        results.sort(key=lambda x: x[1])

        # 상위 N개 chunk만 추출
        top_chunks = [chunk for chunk, _ in results[:top_k]]
        return top_chunks

    def save_index(self, path: str):
        """인덱스 및 chunk 저장"""
        if self.index:
            faiss.write_index(self.index, f"{path}.index")
            with open(f"{path}_chunks.pkl", "wb") as f:
                pickle.dump(self.text_chunks, f)
        
    def load_index(self, path: str):
        """유저 인덱스 및 chunk 로드"""
        if os.path.exists(f"{path}.index") and os.path.exists(f"{path}_chunks.pkl"):
            self.index = faiss.read_index(f"{path}.index")
            with open(f"{path}_chunks.pkl", "rb") as f:
                self.text_chunks = pickle.load(f)
            
    def update_index(self, path: str, new_chunks: list):
        """기존 인덱스를 로드한 후, 새 chunk를 추가하고 저장"""
        if os.path.exists(f"{path}.index") and os.path.exists(f"{path}_chunks.pkl"):
            print(f"[DEBUG] {path} 인덱스 로드 중...")
            self.load_index(path)
        else:
            print(f"[DEBUG] {path} 인덱스가 없어 새로 생성합니다.")
            self.text_chunks = []
            self.index = None

        self.add_chunks(new_chunks)
        self.save_index(path)
    
    def add_chunks(self, new_chunks: list):
        """chunk 누적 추가"""
        self.text_chunks += new_chunks
        new_embeddings = self.model.encode(new_chunks, convert_to_numpy=True)

        if self.index is None:
            print("[DEBUG] index가 없어서 새로 생성합니다.")
            self.index = faiss.IndexFlatL2(new_embeddings.shape[1])

        self.index.add(new_embeddings)
