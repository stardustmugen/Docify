import os
import shutil
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from constants import CHROMA_PERSIST_DIR
import sentence_transformers  
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

persist_directory = CHROMA_PERSIST_DIR

if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

try:
    documents = []
    for root, _, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, file))
                documents.extend(loader.load())

    if not documents:
        print("No documents found.")
        exit()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        texts, embeddings, persist_directory=CHROMA_PERSIST_DIR
    )
    print("Ingestion complete.")

except Exception as e:
    print(f"Error: {e}")
