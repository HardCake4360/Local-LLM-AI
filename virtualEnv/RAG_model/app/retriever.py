#요구사항
"""
pip install sentence-transformers faiss-cpu
"""
# app/retriever.py
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import os
import pickle

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

    def build_index(self, text_chunks: list):
        """문단을 벡터화하고 FAISS 인덱스 생성"""
        self.text_chunks = text_chunks
        embeddings = self.model.encode(text_chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        """질문에 대해 가장 유사한 문단 반환"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in I[0]]

    def save_index(self, path: str):
        """인덱스 + 문단 저장"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}_chunks.pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load_index(self, path: str):
        """저장된 인덱스 + 문단 불러오기"""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}_chunks.pkl", "rb") as f:
            self.text_chunks = pickle.load(f)
