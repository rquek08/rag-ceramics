import streamlit as st
from rag_backend import run_rag_query

st.set_page_config(page_title="RAG Chatbot for Ceramics", layout="wide")
st.title("Ceramics RAG Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

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
                result = run_rag_query(prompt)
                answer = result["answer"]
            except Exception as e:
                answer = f"Error: {str(e)}"
                result = {"retrieved_docs": []}

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show retrieved file names
        if result.get("retrieved_docs"):
            file_names = [
                doc["metadata"].get("source") or doc["metadata"].get("file_name") or "Unknown"
                for doc in result["retrieved_docs"]
            ]
            unique_file_names = list(dict.fromkeys(file_names))

            with st.expander("Files Used for Retrieved Context"):
                st.code("\n".join(unique_file_names), language="markdown")

        # Add LangSmith trace link
        langsmith_link = "https://smith.langchain.com"
        st.markdown(f"[View trace in LangSmith]({langsmith_link})")
