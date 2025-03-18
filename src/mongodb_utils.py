import os
import pymongo
from pymongo import MongoClient
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def connect_to_mongodb(connection_string):
    """Connect to MongoDB using the provided connection string."""
    try:
        client = MongoClient(connection_string)
        print("Connected to MongoDB successfully!")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def get_articles_from_mongodb(client, database_name, collection_name):
    """Retrieve articles from MongoDB."""
    try:
        db = client[database_name]
        collection = db[collection_name]
        articles = list(collection.find({}))
        print(f"Retrieved {len(articles)} articles from MongoDB")
        return articles
    except Exception as e:
        print(f"Error retrieving articles: {e}")
        return []

def convert_to_documents(articles, content_field="content", metadata_fields=None):
    """Convert MongoDB articles to Langchain Document objects."""
    if metadata_fields is None:
        metadata_fields = ["title", "date", "source", "url"]

    documents = []
    for article in articles:
        if content_field in article:
            content = article[content_field]
            # Join list content if necessary
            if isinstance(content, list):
                content = " ".join(content)
            metadata = {field: article.get(field, "") for field in metadata_fields if field in article}
            documents.append(Document(page_content=content, metadata=metadata))
    print(f"Converted {len(documents)} articles to Document objects")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    document_chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(document_chunks)} chunks")
    return document_chunks
