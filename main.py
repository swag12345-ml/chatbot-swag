import streamlit as st  # Streamlit must be imported first

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Chat with Swag AI", page_icon="📝", layout="centered")

import os
import json
import torch
import speech_recognition as sr
from gtts import gTTS
import tempfile
import fitz  # PyMuPDF for text extraction
import easyocr  # GPU-accelerated OCR
from pdf2image import convert_from_path  # Convert PDFs to images
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Optimize GPU performance

def load_groq_api_key():
    """Loads the GROQ API key from config.json"""
    try:
        with open(os.path.join(working_dir, "config.json"), "r") as f:
            return json.load(f).get("GROQ_API_KEY")
    except FileNotFoundError:
        st.error("🚨 config.json not found. Please add your GROQ API key.")
        st.stop()

groq_api_key = load_groq_api_key()
if not groq_api_key:
    st.error("🚨 GROQ_API_KEY is missing. Check your config.json file.")
    st.stop()

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
    
    if DEVICE == "cuda":
        embeddings.model = embeddings.model.to(torch.device("cuda"))
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_chunks = text_splitter.split_text("\n".join(documents))
    return FAISS.from_texts(doc_chunks, embeddings)

def create_chain(vectorstore):
    """Creates the chat chain with optimized retriever settings."""
    if "memory" not in st.session_state:
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

def recognize_speech():
    """Captures user speech and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return "Speech recognition service unavailable."

def text_to_speech(text):
    """Converts chatbot response to speech."""
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        st.session_state.audio_file = temp_audio.name  # Store audio file path

def stop_audio():
    """Stops the currently playing audio."""
    st.session_state.audio_file = None

# Streamlit UI
st.title("🦙 Chat with Swag AI - LLAMA 3.3 (GPU Accelerated)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_extracted_text = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            extracted_text = extract_text_from_pdf(file_path)
            all_extracted_text.extend(extracted_text)
            st.success(f"✅ Extracted text from {uploaded_file.name}")
        except Exception as e:
            st.error(f"⚠️ Error processing {uploaded_file.name}: {e}")
    if all_extracted_text:
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = setup_vectorstore(all_extracted_text)
        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
if "conversation_chain" in st.session_state:
    for message in st.session_state.memory.load_memory_variables({}).get("chat_history", []):
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)
user_input = st.chat_input("Ask Llama...")

if st.button("🎤 Speak"):
    user_input = recognize_speech()
    st.text(f"You said: {user_input}")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    response = st.session_state.conversation_chain.invoke({
        "question": user_input,
        "chat_history": st.session_state.memory.chat_memory.messages
    })
    assistant_response = response.get("answer", "I'm sorry, I couldn't process that.")
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    text_to_speech(assistant_response)
    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file, format="audio/mp3")
    if st.button("⏹ Stop Audio"):
        stop_audio()
    st.session_state.memory.save_context({"input": user_input}, {"output": assistant_response})
