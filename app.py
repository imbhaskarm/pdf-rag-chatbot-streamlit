import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ⚠️ Updated from deprecated langchain_classic → langchain (latest as of 2025)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ⚠️ Updated from deprecated OllamaEmbeddings (requires local Ollama server)
# → HuggingFaceEmbeddings (free, runs locally, no server needed) (latest as of 2025)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)


def create_vector_embeddings(pdf_path: str):
    """Load a PDF, split it into chunks, embed with HuggingFace, store in FAISS."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    # Limit to first 50 pages so embedding build stays fast for demos
    final_documents = text_splitter.split_documents(documents[:50])

    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors


# --- Streamlit UI ---
st.title("PDF RAG Chatbot")
st.caption("Upload a PDF, build the vector index, then ask questions about it.")

upload_col, btn_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

with btn_col:
    embed_btn = st.button("Build Index", use_container_width=True)

if embed_btn:
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        # Save uploaded file to a temp path so PyPDFLoader can read it
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Building vector index... (first run downloads the embedding model ~90 MB)"):
            st.session_state.vectors = create_vector_embeddings(temp_path)
        st.success("Vector index ready.")

user_prompt = st.text_input("Ask a question about the document:")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please upload a PDF and click 'Build Index' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed = time.process_time() - start

        st.write(response["answer"])
        st.caption(f"Response time: {elapsed:.2f}s")

        with st.expander("Source chunks used"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}**")
                st.write(doc.page_content)
                st.write("---")
