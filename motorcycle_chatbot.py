
import streamlit as st
import requests
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
group_passcode = st.sidebar.text_input("Enter Group Passcode", type="password")

if group_passcode != st.secrets["GROUP_PASSCODE"]:
    st.warning("Enter the correct group passcode to continue.")
    st.stop()

# Fetch chat file from Google Drive
@st.cache_data
def load_cleaned_chat():
    file_id = "14zlkLUAeSkCeYxxF3ATSExMqSYf6bDj8"
    direct_link = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(direct_link)
    if response.status_code == 200:
        return response.text
    else:
        return ""

chat_text = load_cleaned_chat()

if not chat_text:
    st.error("Failed to load chat file. Please check the file link.")
    st.stop()

# Prepare documents
docs = [Document(page_content=chat_text)]

# Split into tighter chunks to stay well below token limits
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # smaller chunks
    chunk_overlap=50  # some overlap for context retention
)

chunks = splitter.split_documents(docs)

# Optional: Reduce total chunks if needed to stay under token cap
MAX_TOTAL_CHARS = 200000  # ~250k tokens
current_chars = 0
safe_chunks = []

for doc in chunks:
    if current_chars + len(doc.page_content) > MAX_TOTAL_CHARS:
        break
    safe_chunks.append(doc)
    current_chars += len(doc.page_content)

# Embed and build retriever
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = DocArrayInMemorySearch.from_documents(chunks, embedding=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

# Set up QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
    retriever=retriever
)

# UI
st.title("🏍️ Triumph Owners Group Chatbot")
query = st.text_input("Ask a question about bikes, plans, or issues discussed earlier")

if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
        st.success(answer)
