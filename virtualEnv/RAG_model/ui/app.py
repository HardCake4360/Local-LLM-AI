# ui/streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:5000/ask"

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 RAG 챗봇 (PDF 기반)")
st.markdown("#### 준비된 문서를 바탕으로 질문을 해보세요!")

# 세션 상태로 대화 유지
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요...")
if user_input:
    # 유저 메시지 출력
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("생성 중..."):
            try:
                res = requests.post(API_URL, json={"question": user_input})
                res.raise_for_status()
                answer = res.json().get("answer", "오류: 응답 없음")
            except Exception as e:
                answer = f"❌ 서버 오류: {e}"
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
