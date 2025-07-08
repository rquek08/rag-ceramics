import streamlit as st
from rag_backend import run_rag_query

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG for Ceramics!")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# For storing retrieved docs and scores for each assistant message
if "retrievals" not in st.session_state:
    st.session_state.retrievals = []

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        with st.spinner("Running RAG pipeline..."):
            result = run_rag_query(query)
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        # Add assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
        # Store retrievals for this turn
        st.session_state.retrievals.append({
            "docs": result["retrieved_docs"],
            "scores": result["similarity_scores"]
        })

# Display chat history in order
for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**CeramicsBot:** {msg['content']}")
        # Show retrieved files and scores for this assistant message
        retrieval_idx = idx // 2  # Each user/assistant pair is one retrieval
        if retrieval_idx < len(st.session_state.retrievals):
            docs = st.session_state.retrievals[retrieval_idx]["docs"]
            scores = st.session_state.retrievals[retrieval_idx]["scores"]
            file_lines = []
            for doc, score in zip(docs, scores):
                file_name = doc["metadata"].get("source") or doc["metadata"].get("file_name") or "Unknown"
                file_lines.append(f"{file_name} (Score: {score:.4f})")
            if file_lines:
                st.caption("Files Used for Retrieved Context:")
                st.code("\n".join(file_lines), language="markdown")

langsmith_link = "https://smith.langchain.com"  # Replace if you're capturing a run URL
st.markdown(f"View trace in [LangSmith]({langsmith_link})")
