from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

def ingest_documents():
    documents = []
    
    for fichier in os.listdir("documents"):
        chemin = os.path.join("documents", fichier)
        if fichier.endswith(".pdf"):
            loader = PyPDFLoader(chemin)
        elif fichier.endswith(".txt"):
            loader = TextLoader(chemin, encoding="utf-8")
        else:
            continue
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3.2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    
    print(f"OK : {len(chunks)} chunks indexes avec succes !")

ingest_documents()