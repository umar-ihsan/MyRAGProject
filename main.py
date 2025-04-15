from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from src.mongodb_utils import connect_to_mongodb, get_articles_from_mongodb, convert_to_documents, split_documents
from src.vector_store import initialize_embeddings, load_or_create_faiss_vectorstore
from src.rag import run_rag_system

# Initialize FastAPI
app = FastAPI()


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

# Setup MongoDB and FAISS at startup
def setup_documents():
    connection_string = "mongodb+srv://jamshidjunaid763:JUNAID12345@insightwirecluster.qz5cz.mongodb.net/?retryWrites=true&w=majority&appName=InsightWireCluster"
    database_name = "Scraped-Articles-10"
    collection_name = "Articles2"

    client = connect_to_mongodb(connection_string)
    if client:
        articles = get_articles_from_mongodb(client, database_name, collection_name)
        documents = convert_to_documents(articles)
        document_chunks = split_documents(documents)
        return document_chunks
    else:
        print("MongoDB connection failed. Exiting.")
        return []

print("Initializing document store and vector database...")
document_chunks = setup_documents()
if not document_chunks:
    raise RuntimeError("No documents loaded. Exiting.")

embeddings = initialize_embeddings()
vector_store = load_or_create_faiss_vectorstore(document_chunks, embeddings)

# API request model
class QueryRequest(BaseModel):
    session_id: str  # Unique identifier for the user session
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        session_id = request.session_id

        # Retrieve past conversation history for this session
        if session_id in chat_history:
            past_conversation = " ".join(chat_history[session_id])
        else:
            past_conversation = ""
            chat_history[session_id] = []

        # Append the new query to history
        full_query = f"{past_conversation} {request.query}".strip()
        response = run_rag_system(full_query, vector_store)

        # Store the latest query-response pair
        chat_history[session_id].append(f"User: {request.query}")
        chat_history[session_id].append(f"Bot: {response}")

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn main:app --host 0.0.0.0 --port 8002 --reload
