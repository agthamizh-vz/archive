"""Simple Streamlit frontend for the ViewZen RAG API."""

import os
import requests
import streamlit as st

DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8000")
TOP_K = 5
SUGGESTIONS = [
    "What are the standard roles in ViewZen?",
    "How do I create a dashboard?",
    "What is Appverse?",
    "How do permissions work?",
]


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for src in sources:
            label = src.get("title") or src.get("section") or src.get("url")
            st.markdown(f"- [{label}]({src.get('url', '')})")


def ask_api(api_base: str, prompt: str, top_k: int) -> tuple[str, list[dict]]:
    res = requests.post(
        f"{api_base}/ask",
        json={"query": prompt, "top_k": top_k},
        timeout=120,
    )
    res.raise_for_status()
    data = res.json()
    return data.get("answer", ""), data.get("sources", [])


st.set_page_config(page_title="ViewZen Assistant", page_icon="AI", layout="centered")
st.title("ViewZen AI Assistant")
st.caption("Chat with your ViewZen documentation knowledge base")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    api_base = DEFAULT_API

    st.subheader("History")
    user_messages = [m["content"] for m in st.session_state.messages if m.get("role") == "user"]
    if user_messages:
        for i, text in enumerate(user_messages[-10:], 1):
            preview = text if len(text) <= 60 else text[:60] + "..."
            st.caption(f"{i}. {preview}")
    else:
        st.caption("No questions yet.")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Backend Status")
    try:
        health = requests.get(f"{api_base}/health", timeout=8).json()
        st.success("Connected")
        st.write(f"Model: `{health.get('model', 'N/A')}`")
        st.write(f"Vectors: `{health.get('vector_store_count', 0)}`")
    except Exception as exc:
        st.error(f"Cannot connect: {exc}")

if len(st.session_state.messages) == 0:
    st.markdown("#### Try one of these")
    c1, c2 = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        target = c1 if i % 2 == 0 else c2
        if target.button(suggestion, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, sources = ask_api(api_base, suggestion, TOP_K)
                    except Exception as exc:
                        answer, sources = f"Error: {exc}", []
                st.markdown(answer)
                render_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

prompt = st.chat_input("Ask about ViewZen...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask_api(api_base, prompt, TOP_K)
            except Exception as exc:
                answer, sources = f"Error: {exc}", []
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
