import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# Path for persistent FAISS vector store
FAISS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index')

def initialize_embeddings(model="sentence-transformers/all-MiniLM-L6-v2"):
    """Initialize and return the HuggingFaceEmbeddings model."""
    return HuggingFaceEmbeddings(model_name=model)

def create_faiss_vectorstore(documents, embeddings, save_path=FAISS_PATH):
    """Create a FAISS vector store from documents and save it."""
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(save_path)
    print(f"Vector store created and saved to {save_path}")
    return vector_store

def load_faiss_vectorstore(embeddings, load_path=FAISS_PATH):
    """Load a FAISS vector store from disk."""
    vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)

    print(f"Vector store loaded from {load_path}")
    return vector_store

def create_chroma_vectorstore(documents, embeddings, persist_directory="chroma_db"):
    """Create a Chroma vector store from documents and persist it."""
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    print(f"Chroma vector store created and persisted to {persist_directory}")
    return vector_store

def load_or_create_faiss_vectorstore(documents, embeddings, load_path=FAISS_PATH):
    """Load existing FAISS vector store if it exists, else create a new one."""
    if os.path.exists(load_path):
        print("Loading persistent FAISS vector store...")
        vector_store = load_faiss_vectorstore(embeddings, load_path)
    else:
        print("Creating new FAISS vector store...")
        vector_store = create_faiss_vectorstore(documents, embeddings, load_path)
    return vector_store

# Uncomment the block below to reset the persistent vector store:
'''
import shutil
if os.path.exists(FAISS_PATH):
    shutil.rmtree(FAISS_PATH)
    print("Persistent vector store removed. A new one will be created on next run.")
'''
