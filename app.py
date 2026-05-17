# app.py
import streamlit as st
import tempfile
import os
from rag_engine import load_and_index_pdf, create_vector_store, load_vector_store, get_answer

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.developer {
    text-align: center;
    color: #764ba2;
    font-size: 0.85rem;
    font-weight: 600;
}
.chat-question {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
}
.chat-answer {
    background: var(--background-color);
    border: 1px solid #667eea;
    border-left: 4px solid #667eea;
    color: var(--text-color) !important;
    padding: 1rem;
    border-radius: 0 12px 12px 0;
    margin: 0.5rem 0;
}
.source-box {
    background: var(--background-color);
    border: 1px solid #667eea;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    color: var(--text-color) !important;
    margin-top: 0.3rem;
}
.footer {
    text-align: center;
    font-size: 0.8rem;
    margin-top: 2rem;
    color: var(--text-color) !important;
    opacity: 0.7;
}
p, div, span {
    color: var(--text-color);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">📄 PDF Q&A Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="developer">👨‍💻 Developed by Muhammad Arif</p>', unsafe_allow_html=True)
st.divider()

# Session state initialize
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

# Sidebar
with st.sidebar:
    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("⚡ Process PDFs", use_container_width=True):
            with st.spinner("Reading and indexing PDFs..."):
                tmp_paths = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_paths.append(tmp.name)

                chunks = load_and_index_pdf(tmp_paths)
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.pdf_names = [f.name for f in uploaded_files]
                st.session_state.chat_history = []

            st.success(f"✅ {len(uploaded_files)} PDF(s) indexed!")
            st.info(f"📊 {len(chunks)} chunks created")

    # Show uploaded PDFs
    if st.session_state.pdf_names:
        st.divider()
        st.markdown("**📚 Loaded PDFs:**")
        for name in st.session_state.pdf_names:
            st.markdown(f"- 📄 {name}")

    # Clear button
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Main chat area
if st.session_state.vector_store is None:
    st.info("👈 Please upload and process a PDF from the sidebar to get started!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📤 Step 1**
        Upload your PDF files from the sidebar
        """)
    with col2:
        st.markdown("""
        **⚡ Step 2**
        Click 'Process PDFs' button
        """)
    with col3:
        st.markdown("""
        **💬 Step 3**
        Ask any question about your PDF!
        """)
else:
    # Show chat history
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="chat-question">🧑 {chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-answer">🤖 {chat["answer"]}</div>', unsafe_allow_html=True)

        # Show sources
        with st.expander("📍 View Sources"):
            for doc in chat["sources"]:
                page = doc.metadata.get("page", "?")
                source = doc.metadata.get("source_file", "PDF")
                st.markdown(f'<div class="source-box">📄 {source} — Page {int(page)+1}</div>', unsafe_allow_html=True)
                st.caption(doc.page_content[:200] + "...")

    # Question input
    question = st.chat_input("Ask a question about your PDF...")

    if question:
        with st.spinner("🤔 Thinking..."):
            answer, source_docs = get_answer(
                question,
                st.session_state.vector_store,
                st.session_state.chat_history
            )

        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": source_docs
        })

        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align:center; color:#adb5bd; font-size:0.8rem;'>
    Built with LangChain + FAISS + Gemini AI | Made with ❤️ by <b>Muhammad Arif</b>
</div>
""", unsafe_allow_html=True)