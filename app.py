import streamlit as st
from rag_backend import run_rag_query
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="RAG Chatbot for Ceramics", layout="wide")
st.title("RAG Chatbot for Ceramics")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input
user_input = st.chat_input("Ask something about ceramics...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run RAG
    with st.spinner("Thinking..."):
        result = run_rag_query(user_input)

    ai_response = result["answer"]

    # Store in history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=ai_response))

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

        # Optional: show retrieved sources
        file_names = []
        for doc in result["retrieved_docs"]:
            file_name = doc["metadata"].get("source") or doc["metadata"].get("file_name") or "Unknown"
            file_names.append(file_name)
        unique_file_names = list(dict.fromkeys(file_names))

        with st.expander("📄 Files Used for Context"):
            st.code("\n".join(unique_file_names), language="markdown")

        langsmith_link = "https://smith.langchain.com"
        st.markdown(f"View trace in [LangSmith]({langsmith_link})")
