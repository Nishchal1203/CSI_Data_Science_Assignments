import streamlit as st
from scripts.query_and_generate import retrieve, generate_answer

st.set_page_config(page_title="ISO/IEC 2017 Q&A Bot", layout="centered")

st.title("📘 ISO/IEC 2017 RAG Q&A Chatbot")

query = st.text_input("🔍 Ask your question:")
if query:
    with st.spinner("Retrieving and generating answer..."):
        ctx = retrieve(query)
        answer = generate_answer(query, ctx)

        st.markdown("### 💡 Answer:")
        st.write(answer)

        with st.expander("📚 Retrieved Context"):
            for c in ctx:
                st.write(c)
                st.markdown("---")
