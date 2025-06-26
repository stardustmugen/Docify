import streamlit as st
import os
import base64
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_PERSIST_DIR
from streamlit_chat import message
import sentence_transformers  

device = torch.device("cpu")

checkpoint = "./models/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

@st.cache_resource(show_spinner=False)
def data_ingestion():
    try:
        for root, dirs, files in os.walk("docs"):
            for file in files:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(root, file))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_PERSIST_DIR)
        return True
    except Exception as e:
        st.error(f"Error during data ingestion: {e}")
        return False

@st.cache_resource(show_spinner=False)
def llm_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource(show_spinner=False)
def qa_llm():
    llm = llm_pipeline()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    qa = qa_llm()
    result = qa(instruction)
    return result['result']

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=f"{i}_user")
        message(history["generated"][i], key=f"{i}")

def main():
    st.markdown("""
        <style>
            .main-header {
                text-align: center;
                background-color:rgb(8, 60, 81);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-header'><h1>DociFy - Converse with Your PDF 📄</h1></div>", unsafe_allow_html=True)

    st.sidebar.markdown("""
        <style>
            .docify-title {
                font-size: 42px;
                font-weight: bold;
                text-align: center;
                color:white;
                margin-top: -72px;
                margin-left: -160px;
            }
        </style>
        <div class='docify-title'>DociFy</div>
    """, unsafe_allow_html=True)

    st.sidebar.title("📖 Instructions")
    st.sidebar.markdown("""
        - Upload a PDF document to begin.
        - Embeddings will be created.
        - Ask questions from the content!
    """)

    if st.sidebar.button("🔁 Clear All Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Cache cleared. You can reupload a document now.")

    if not os.path.exists("docs"):
        os.makedirs("docs")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        filepath = os.path.join("docs", uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded: {uploaded_file.name}")
        displayPDF(filepath)

        with st.spinner("Embedding your document..."):
            success = data_ingestion()

        if success:
            st.success("Embeddings created successfully!")

            if "generated" not in st.session_state:
                st.session_state["generated"] = ["Ask me anything from your PDF!"]
                st.session_state["past"] = ["Hello"]

            query = st.text_input("🔍 Enter your question:")
            if query:
                answer = process_answer({"query": query})
                st.session_state["past"].insert(0, query)
                st.session_state["generated"].insert(0, answer)

            display_conversation(st.session_state)

if __name__ == "__main__":
    main()
