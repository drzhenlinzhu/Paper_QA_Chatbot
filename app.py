import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = FAISS.load_local("astronomy_vectorstore", embeddings, allow_dangerous_deserialization=True)

# Build QA chain
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":4})
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    chain_type="stuff",
    retriever=retriever
)

# Streamlit app
st.title("Astronomy Paper QA Chatbot")

query = st.text_input("Ask a question about the paper:")

if query:
    answer = qa.run(query)
    st.write(answer)

