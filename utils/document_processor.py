import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings

def initialize_vector_store(embeddings, index_name="ai-medi"):
    """Load existing vector store from Pinecone."""
    try:
        print("Loading existing vector store...")
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        print("Successfully loaded existing vector store")
        return docsearch
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        raise

def initialize_retriever(docsearch, k=3):
    """Initialize and return the retriever."""
    return docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def process_documents():
    """Initialize vector store and retriever using existing Pinecone index."""
    try:
        # Initialize embeddings
        print("Initializing embeddings...")
        embeddings = download_embeddings()
        print("Embeddings initialized successfully")
        
        # Load existing vector store
        docsearch = initialize_vector_store(embeddings)
        
        # Initialize retriever
        print("Initializing retriever...")
        retriever = initialize_retriever(docsearch)
        print("Retriever initialized successfully")
        
        return retriever, embeddings
    
    except Exception as e:
        print(f"Error in process_documents: {str(e)}")
        raise 