import streamlit as st
from rag_backend import run_rag_query

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG for Ceramics!")

query = st.text_input("Ask a question:")

if st.button("Submit"):
    if query:
        with st.spinner("Running RAG pipeline..."):
            result = run_rag_query(query)

        st.subheader("Answer:")
        st.write(result["answer"])

        # Show the file names of the retrieved documents
        file_names = []
        for doc in result["retrieved_docs"]:
            # Try common metadata keys for file name
            file_name = doc["metadata"].get("source") or doc["metadata"].get("file_name") or "Unknown"
            file_names.append(file_name)
        unique_file_names = list(dict.fromkeys(file_names))  # Remove duplicates, preserve order

        st.subheader("Files Used for Retrieved Context:")
        st.code("\n".join(unique_file_names), language="markdown")

        langsmith_link = "https://smith.langchain.com"  # Replace if you're capturing a run URL
        st.markdown(f"View trace in [LangSmith]({langsmith_link})")
    else:
        st.warning("Please enter a question.")