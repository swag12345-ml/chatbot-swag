import streamlit as st  # Streamlit must be imported first

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Chat with Doc", page_icon="📝", layout="centered")

import os
import json
import torch
from dotenv import load_dotenv
import fitz  # PyMuPDF for text extraction
import easyocr  # GPU-accelerated OCR
from pdf2image import convert_from_path  # Convert PDFs to images
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Fixed import
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np

# Load environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

# Load GROQ API Key from config.json
def load_groq_api_key():
    """Loads the GROQ API key from config.json"""
    config_path = os.path.join(working_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    return None

groq_api_key = load_groq_api_key()
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in config.json.")

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Initialize EasyOCR with GPU support
reader = easyocr.Reader(["en"], gpu=True)

def extract_text_from_pdf(file_path):
    """Extracts text from PDFs using PyMuPDF, falls back to GPU-based OCR if needed."""
    doc = fitz.open(file_path)
    text_list = [page.get_text("text") for page in doc if page.get_text("text").strip()]
    doc.close()
    return text_list if text_list else extract_text_from_images(file_path)

def extract_text_from_images(pdf_path):
    """Extracts text from image-based PDFs using GPU-accelerated EasyOCR."""
    images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=5)
    return ["\n".join(reader.readtext(np.array(img), detail=0)) for img in images]

def setup_vectorstore(documents):
    """Creates a FAISS vector store using Hugging Face embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Move model to GPU if available
    if DEVICE == "cuda":
        embeddings.model = embeddings.model.to(torch.device("cuda"))

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))

    return FAISS.from_texts(doc_chunks, embeddings)

def create_chain(vectorstore):
    """Creates the chat chain with optimized retriever settings."""
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=st.session_state.memory,
        verbose=False
    )

# Streamlit UI
st.title("🦙 Chat with Doc - LLAMA 3.3 (GPU Accelerated)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

if uploaded_file:
    file_path = os.path.join(working_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        extracted_text = extract_text_from_pdf(file_path)
        st.success("✅ Text extracted successfully!")

        # Reset chat memory when a new document is uploaded
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        st.session_state.vectorstore = setup_vectorstore(extracted_text)
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

if "conversation_chain" in st.session_state:
    for message in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

user_input = st.chat_input("Ask Llama...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Clear memory to prevent previous responses from persisting incorrectly
    st.session_state.memory.chat_memory.clear()

    response = st.session_state.conversation_chain.invoke({
        "question": user_input,
        "chat_history": []  # Reset chat history to avoid repeated answers
    })

    assistant_response = response.get("answer", "I'm sorry, I couldn't process that.")

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Save only the latest question-answer pair
    st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})
