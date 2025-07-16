import streamlit as st
import openai
import pandas as pd
import os
import tempfile
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="Triumph 400 Riders Bot", page_icon="🏍️")
st.title("🏍️ Triumph 400 Riders - Chatbot")

# ------------------------------
# 🔐 Load secrets securely
# ------------------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]
group_passcode = st.sidebar.text_input("Enter Group Passcode", type="password")

if group_passcode != st.secrets["GROUP_PASSCODE"]:
    st.warning("Enter the correct group passcode to continue.")
    st.stop()

uploaded_file = st.sidebar.file_uploader("Upload Cleaned WhatsApp Chat (.txt)", type="txt")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

if not uploaded_file:
    st.info("Upload your cleaned chat file to get started.")
    st.stop()

# Save uploaded file temporarily
with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    tmp_path = tmp_file.name

# Load chat data
loader = TextLoader(tmp_path)
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create Chroma DB
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = DocArrayInMemorySearch.from_documents(chunks, embedding=embeddings)

# Setup retrieval + GPT
retriever = db.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
    retriever=retriever
)

# Chat input
query = st.text_input("Ask a question about motorcycles, riding gear, issues, etc.")

if query:
    with st.spinner("Searching the chat..."):
        result = qa_chain.run(query)
        st.success(result)