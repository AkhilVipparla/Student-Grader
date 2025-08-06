import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

#Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

#Build or Load Vector DB
@st.cache_resource
def get_vector_store(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_text(pdf_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

#Get relevant context from DB
def retrieve_context(query, vector_store, k=5):
    docs = vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

#UI
st.title("Answer Retriever")

uploaded_pdf = st.file_uploader("Upload Reference Material (PDF)", type="pdf")
question = st.text_input("Enter Question")

if uploaded_pdf and question:
    with open("reference_material.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf("reference_material.pdf")
        vector_store = get_vector_store(pdf_text)
        context = retrieve_context(question, vector_store)

    # Save context to text file
    with open("retrieved_context.txt", "w") as f:
        f.write(context.encode("cp1252", errors="replace").decode("cp1252"))


    st.success("✅ Context saved to retrieved_context.txt")
    st.text_area("Retrieved Context", context, height=300)


