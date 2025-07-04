import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# PDF parsing
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text
pdf_path = "suzaku_2009.pdf"
text = extract_text_from_pdf(pdf_path)

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.create_documents([text])

# Embed
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = FAISS.from_documents(docs, embeddings)

# Save
db.save_local("astronomy_vectorstore")
print("Vector store saved.")

