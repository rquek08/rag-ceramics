import streamlit as st
from rag_backend import run_rag_query

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG for Ceramics!")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        with st.spinner("Running RAG pipeline..."):
            result = run_rag_query(query)
        # Append user query and bot answer to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**CeramicsBot:** {msg['content']}")

# Optionally, show retrieved files for the last answer
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
    # Only show for the last bot response
    if "result" in locals():
        file_names = []
        for doc in result["retrieved_docs"]:
            file_name = doc["metadata"].get("source") or doc["metadata"].get("file_name") or "Unknown"
            file_names.append(file_name)
        unique_file_names = list(dict.fromkeys(file_names))
        st.subheader("Files Used for Retrieved Context:")
        st.code("\n".join(unique_file_names), language="markdown")

        langsmith_link = "https://smith.langchain.com"  # Replace if you're capturing a run URL
        st.markdown(f"View trace in [LangSmith]({langsmith_link})")
