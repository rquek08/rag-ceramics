# Import necessary libraries
import os
from typing_extensions import List, TypedDict
from huggingface_hub import hf_hub_download
import pickle
import faiss
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from langgraph.graph import MessagesState
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain.chat_models import init_chat_model

# === Load API keys from Streamlit secrets ===
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langsmith"]["api_key"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["langsmith"]["project"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# === Initialize LLM and embeddings ===
llm = init_chat_model("gpt-4.1-mini", model_provider="openai")
embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")

# === Load FAISS index and docstore from Hugging Face ===
REPO_ID = "rachq/rag-ceramics"  

faiss_index_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="index.faiss",
    repo_type="dataset",
    use_auth_token=st.secrets["huggingface"]["token"]
)

docstore_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="index.pkl",
    repo_type="dataset",
    use_auth_token=st.secrets["huggingface"]["token"]
)

# === Load index and docstore ===
index = faiss.read_index(faiss_index_path)
with open(docstore_path, "rb") as f:
    docstore_data, index_to_docstore_id = pickle.load(f)

# Ensure docstore is an InMemoryDocstore
if not isinstance(docstore_data, InMemoryDocstore):
    docstore = InMemoryDocstore(docstore_data)
else:
    docstore = docstore_data

vector_store = FAISS(embedding_fn.embed_query, index, docstore, index_to_docstore_id)

# === LangGraph State ===
class State(TypedDict):
    messages: List[BaseMessage]
    context: List[Document]

# === Retrieval function ===
def retrieve(state: State) -> dict:
    user_msg = next(m for m in reversed(state["messages"]) if m.type == "human")
    query = user_msg.content
    results_with_scores = vector_store.similarity_search_with_score(query, k=8)

    retrieved_docs = [doc for doc, score in results_with_scores]
    similarity_scores = [score for doc, score in results_with_scores]

    return {
        "messages": state["messages"],
        "context": retrieved_docs,
        "similarity_scores": similarity_scores
    }

# === Generation function ===
def generate(state: State) -> dict:
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the context below to answer the question.\n\n"
        f"Context:\n{docs_content}"
    )

    conversation = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(conversation)

    updated_messages = state["messages"] + [response]

    return {
        "messages": updated_messages,
        "context": state["context"]
    }

# === Build LangGraph ===
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.set_finish_point("generate")
graph = builder.compile()

# === Streamlit integration ===
chat_history = []

def run_rag_query(user_input, chat_history):
    chat_history.append(HumanMessage(content=user_input))
    state = {
        "messages": chat_history,
        "context": []
    }

    retrieved_docs = []
    similarity_scores = []
    result_msg = ""

    for step in graph.stream(state, stream_mode="values"):
        if "similarity_scores" in step:
            similarity_scores = step["similarity_scores"]
        if "context" in step:
            retrieved_docs = step["context"]
        if "messages" in step and step["messages"][-1].type == "ai":
            result_msg = step["messages"][-1].content
            chat_history.append(step["messages"][-1])

    return {
        "answer": result_msg,
        "retrieved_docs": [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs],
        "similarity_scores": similarity_scores
    }
