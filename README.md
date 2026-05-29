# PDF RAG Chatbot with Streamlit

A Streamlit app that lets you upload any PDF, build a FAISS vector index from it, and ask questions answered by a Groq LLM using retrieved context.

Built while learning how LangChain's retrieval chain connects a document store to an LLM — and how to replace Ollama-based embeddings with a free HuggingFace model that runs entirely locally.

---

## How It Works

```
Upload PDF
    |
    v
[PyPDFLoader]                  -- loads all pages
    |
    v
[RecursiveCharacterTextSplitter]  -- splits into 1000-char chunks with 200-char overlap
    |
    v
[HuggingFaceEmbeddings]        -- embeds each chunk (all-MiniLM-L6-v2, runs on CPU)
    |
    v
[FAISS vectorstore]            -- stores embeddings in memory
    |
    v
User types question
    |
    v
[create_retrieval_chain]       -- retrieves top-k relevant chunks
    |
    v
[create_stuff_documents_chain] -- injects chunks into prompt as context
    |
    v
[Groq LLM]                    -- generates answer grounded in retrieved context
```

---

## Setup

```bash
git clone https://github.com/imbhaskarm/pdf-rag-chatbot-streamlit.git
cd pdf-rag-chatbot-streamlit
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Add your Groq API key to `.env`. Get a free key at https://console.groq.com

---

## Run

```bash
streamlit run app.py
```

1. Upload any PDF using the file uploader
2. Click **Build Index** (first run downloads the embedding model ~90 MB — cached after that)
3. Type your question and press Enter

---

## Bugs Fixed vs Original

| Bug | Original | Fix |
|---|---|---|
| Non-existent package | `from langchain_classic.chains import ...` | `from langchain.chains import ...` |
| Embeddings | `OllamaEmbeddings` (requires Ollama server running) | `HuggingFaceEmbeddings` (free, no server) |
| Hardcoded PDF path | `PyPDFLoader("attention.pdf")` (fails on clone) | Streamlit file uploader → temp file path |

---

## What I Learned

- `create_stuff_documents_chain` handles the "stuff all docs into the prompt" pattern correctly; manually formatting a list of Document objects as a string always breaks because the template expects a string, not a list
- FAISS `from_documents()` is an in-memory index — it resets every app restart, which is fine for demos but means you'd need `save_local()` / `load_local()` for production persistence
- LangChain's `retrieval_chain.invoke()` returns both `answer` and `context`, so you can always show the user which source chunks were used — useful for debugging hallucinations

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | Web UI |
| LangChain | RAG chain, document loading, text splitting |
| Groq (Llama 3.3 70B) | LLM inference |
| FAISS | Vector similarity search |
| HuggingFace sentence-transformers | Free local embeddings |

---

## GitHub Topics

`streamlit` `rag` `langchain` `faiss` `groq`
