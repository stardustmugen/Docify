# Docify
# 📄 Docify – Converse with Your PDF Using AI

**Docify** is a lightweight AI-powered web application that allows users to **upload PDF documents** and interact with them using **natural language**. Instead of manually searching through documents, users can ask questions and receive intelligent, context-aware answers — all processed **locally** with no internet or API dependency.

---

## 🔧 Tech Stack

- **Frontend**: Streamlit (for chatbot UI)
- **Text Extraction**: LangChain + PyPDFLoader
- **Embedding Model**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: ChromaDB
- **LLM**: LaMini-T5-738M (locally hosted)
- **Retrieval Logic**: LangChain `RetrievalQA`

---

## 🚀 Features

- 📄 Upload and interact with any PDF document
- 🔍 Fast semantic search using vector similarity
- 💬 Ask questions in plain English and get contextual answers
- 🔒 Fully local — your data never leaves your machine
- ⚡ Lightweight, runs even on modest hardware

---

## 🧠 How It Works

1. **Upload PDF**: Text is extracted using PyPDFLoader  
2. **Preprocessing**: Content is cleaned and chunked  
3. **Embedding**: Each chunk is converted into semantic vectors  
4. **Storage**: Stored in ChromaDB for efficient similarity search  
5. **Querying**: User’s query is embedded and compared to document chunks  
6. **Answer Generation**: Top matches are passed to LaMini-T5 to generate a response

---

## 💻 Installation
git clone https://github.com/yourusername/docify.git
cd docify

- Download the model from hggingface.
- Create 2 folders inside docify (models, db)
- Copy the the model into the models folder 

# Set up environment
python -m venv docify_env
docify_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run chatbot_app.py
