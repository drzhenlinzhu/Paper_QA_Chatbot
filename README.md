# Paper_QA_Chatbot

An end-to-end GenAI Retrieval-Augmented Generation (RAG) chatbot built to assist with academic paper comprehension, powered by OpenAI embeddings and GPT-4o, with a responsive Streamlit web interface.

## Overview

This project implements PDF parsing, text chunking, and embedding generation using the OpenAI API (`text-embedding-3-large`), with semantic indexing stored in FAISS for fast retrieval. A LangChain `RetrievalQA` chain backed by `gpt-4o` improves the accuracy of responses to user queries, all served through an interactive Streamlit chat interface.

**Key features:**
- **Dynamic paper uploads** — upload any PDF directly from the browser; no rebuild step required
- **Multi-paper knowledge base** — upload multiple PDFs and query across all of them simultaneously
- **Persistent chat history** — full conversation thread displayed within a session
- **Pre-built vectorstore fallback** — loads `astronomy_vectorstore/` on startup if present, so the app works out of the box

## Project Structure

```
Paper_QA_Chatbot/
├── app.py                 # Streamlit chatbot app (upload + chat UI)
├── build_vectorstore.py   # One-time script to pre-build a FAISS index from a PDF
├── chatbot.ipynb          # Jupyter notebook walkthrough
├── example.pdf            # Sample astronomy paper
├── astronomy_vectorstore/ # Optional pre-built FAISS index
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.8+
- An OpenAI API key

## Running the Project

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Step 3 — Launch the chatbot

```bash
streamlit run app.py
```

Open the URL shown in your terminal (default: http://localhost:8501).

### Step 4 — Upload papers and start querying

1. Use the **sidebar** to upload one or more PDF files.
2. Type your question in the **chat box** at the bottom of the page.
3. The app retrieves the most relevant passages and answers via GPT-4o.
4. Use **"Clear chat history"** in the sidebar to start a fresh conversation.

> If `astronomy_vectorstore/` is present, it is loaded automatically on startup so you can query straight away without uploading anything.

## How It Works

1. **Upload** — when a PDF is uploaded, `app.py` extracts text with PyMuPDF (`fitz`), splits it into 500-character chunks (100-character overlap), embeds each chunk with `text-embedding-3-large`, and merges the result into the in-memory FAISS index.
2. **Session state** — the vectorstore, QA chain, uploaded paper list, and chat history are stored in `st.session_state` so nothing is recomputed on Streamlit reruns.
3. **Retrieval** — at query time, the top-4 most similar chunks are retrieved and passed to `gpt-4o` via a LangChain `RetrievalQA` chain.
4. **Chat UI** — responses are displayed as chat bubbles (`st.chat_message`) and the full conversation history is replayed on each rerun.

## Pre-building a Vectorstore (Optional)

`build_vectorstore.py` is a standalone script for pre-processing a single PDF and saving the FAISS index to disk. Use it if you want to ship a ready-made knowledge base:

```bash
python build_vectorstore.py
```

This reads `example.pdf`, embeds it, and saves the index to `astronomy_vectorstore/`. The app will load it automatically on next startup.
