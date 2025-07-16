import streamlit as st
import openai
import pandas as pd
import os
import tempfile
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="Triumph 400 Riders Bot", page_icon="🏍️")
st.title("🏍️ Triumph 400 Riders - Chatbot")

group_passcode = st.sidebar.text_input("Enter Group Passcode", type="password")
if group_passcode != "speed400":
    st.warning("Enter the correct group passcode to continue.")
    st.stop()

# Sidebar: OpenAI API key input
openai_api_key = "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"

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
db = Chroma.from_documents(chunks, embedding=embeddings)

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