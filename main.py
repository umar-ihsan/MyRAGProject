from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from src.mongodb_utils import connect_to_mongodb, get_articles_from_mongodb, convert_to_documents, split_documents
from src.vector_store import initialize_embeddings, load_or_create_faiss_vectorstore
from src.rag import run_rag_system
from config import (
    MONGODB_CONNECTION_STRING,
    MONGODB_DATABASE_NAME,
    MONGODB_COLLECTION_NAME,
    EMBEDDING_MODEL,
    API_HOST,
    API_PORT,
    API_DEBUG
)
import os

# Initialize FastAPI
app = FastAPI()

# Check if running on Vercel
is_vercel = os.environ.get('VERCEL') == '1'

# Allow all origins for testing (replace "*" with specific domains for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["https://your-frontend.com"] for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Store chat history per user (uses session_id as key)
chat_history: Dict[str, List[str]] = {}

# Global vector store for reuse in serverless environment
vector_store = None

# Setup MongoDB and FAISS at startup
def setup_documents():
    client = connect_to_mongodb(MONGODB_CONNECTION_STRING)
    if client:
        articles = get_articles_from_mongodb(client, MONGODB_DATABASE_NAME, MONGODB_COLLECTION_NAME)
        documents = convert_to_documents(articles)
        document_chunks = split_documents(documents)
        return document_chunks
    else:
        print("MongoDB connection failed. Exiting.")
        return []

# Initialize embeddings and vector store - lazy loading for serverless
def get_vector_store():
    global vector_store
    if vector_store is None:
        print("Initializing document store and vector database...")
        document_chunks = setup_documents()
        if not document_chunks:
            raise RuntimeError("No documents loaded. Exiting.")
        
        embeddings = initialize_embeddings(EMBEDDING_MODEL)
        vector_store = load_or_create_faiss_vectorstore(document_chunks, embeddings)
    return vector_store

# Only initialize immediately if not on Vercel (for local dev)
if not is_vercel:
    print("Local environment detected, initializing...")
    vector_store = get_vector_store()

# API request model
class QueryRequest(BaseModel):
    session_id: str  # Unique identifier for the user session
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        # Get vector store (lazy load on first request in serverless)
        vs = get_vector_store()
        
        session_id = request.session_id

        # Retrieve past conversation history for this session
        if session_id in chat_history:
            past_conversation = " ".join(chat_history[session_id])
        else:
            past_conversation = ""
            chat_history[session_id] = []

        # Append the new query to history
        full_query = f"{past_conversation} {request.query}".strip()
        response = run_rag_system(full_query, vs)

        # Store the latest query-response pair
        chat_history[session_id].append(f"User: {request.query}")
        chat_history[session_id].append(f"Bot: {response}")

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "RAG API is running! Send POST requests to /query endpoint."}

# Only needed for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=int(API_PORT), reload=API_DEBUG)
