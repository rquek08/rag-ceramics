import streamlit as st
from rag_backend import run_rag_query

st.set_page_config(page_title="RAG Chatbot for Ceramics", layout="wide")
st.title("Ceramics RAG Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about ceramics..."):
    # Append user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run RAG query
    with st.chat_message("assistant"):
        with st.spinner("Retrieving information..."):
            try:
                # Pass chat history to your backend
                result = run_rag_query(prompt, st.session_state.chat_history)
                answer = result["answer"]
            except Exception as e:
                answer = f"Error: {str(e)}"
                result = {"retrieved_docs": []}

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show retrieved file names and retrieved chunks
        if result.get("retrieved_docs"):
            with st.expander("Files and Chunks Used for Retrieved Context"):
                for doc in result["retrieved_docs"]:
                    file_name = doc["metadata"].get("source") or doc["metadata"].get("file_name") or "Unknown"
                    chunk = doc.get("page_content", "") if hasattr(doc, "page_content") else doc.get("content", "")
                    st.write(f"**File:** {file_name}")
                    st.code(chunk)
