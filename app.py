import io
import fitz
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

EMBEDDINGS_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"
VECTORSTORE_PATH = "astronomy_vectorstore"


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDINGS_MODEL)


def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)


def build_vectorstore_from_text(text: str, embeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embeddings)


def make_qa_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=CHAT_MODEL),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )


# ── Session state initialisation ────────────────────────────────────────────
if "embeddings" not in st.session_state:
    st.session_state.embeddings = get_embeddings()

if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            st.session_state.embeddings,
            allow_dangerous_deserialization=True,
        )
        st.session_state.uploaded_papers = ["astronomy_vectorstore (pre-built)"]
    except Exception:
        st.session_state.vectorstore = None
        st.session_state.uploaded_papers = []

if "qa_chain" not in st.session_state:
    if st.session_state.vectorstore is not None:
        st.session_state.qa_chain = make_qa_chain(st.session_state.vectorstore)
    else:
        st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"|"assistant", "content": str}


# ── Sidebar: upload & paper list ─────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.uploaded_papers
        ]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} new paper(s)…"):
                for f in new_files:
                    text = extract_text(f.read())
                    new_vs = build_vectorstore_from_text(text, st.session_state.embeddings)
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = new_vs
                    else:
                        st.session_state.vectorstore.merge_from(new_vs)
                    st.session_state.uploaded_papers.append(f.name)
            st.session_state.qa_chain = make_qa_chain(st.session_state.vectorstore)
            st.success(f"Added {len(new_files)} paper(s) to the knowledge base.")

    st.divider()
    st.subheader("Knowledge Base")
    if st.session_state.uploaded_papers:
        for name in st.session_state.uploaded_papers:
            st.markdown(f"- {name}")
    else:
        st.info("No papers loaded yet. Upload a PDF to get started.")

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()


# ── Main chat interface ───────────────────────────────────────────────────────
st.title("Paper QA Chatbot")

# Replay existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
if prompt := st.chat_input("Ask a question about the uploaded paper(s)…"):
    if st.session_state.qa_chain is None:
        st.warning("Please upload at least one PDF before asking a question.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = st.session_state.qa_chain.run(prompt)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
