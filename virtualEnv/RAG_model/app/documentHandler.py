#요구사항
"""
pip install pymupdf     :pdf 텍스트 추출 라이브러리
"""

import fitz  # PyMuPDF
import os

def extract_text_from_pdf(file_path: str) -> str:
    """PDF 파일에서 전체 텍스트 추출"""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_text_to_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """긴 텍스트를 문단(또는 고정 크기 블록) 단위로 나누기"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # 겹치게 하여 문맥 유지
    return chunks

def split_by_paragraph(text: str):
    return [p.strip() for p in text.split('\n \n') if len(p.strip()) > 50]


if __name__ == "__main__":
    text = extract_text_from_pdf("TetoPrompt.pdf")
    chunks = split_by_paragraph(text)

    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")
