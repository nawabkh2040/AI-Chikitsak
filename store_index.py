from src.helper import load_pdf_files, text_splitter, download_embeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer


import os 
PINECORE_API_KEY = os.getenv("PINECORN")
os.environ["PINECONE_API_KEY"] = PINECORE_API_KEY

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

extracted_data = load_pdf_files(data_path="Data/")
text_chunks = text_splitter(extracted_data)
print("Length of the text chunks: ", len(text_chunks))

embeddings = download_embeddings()

pc = Pinecone(api_key=PINECORE_API_KEY)
index_name = "ai-medi"
pc.create_index(
  name=index_name,
  dimension=384,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
)


os.environ["PINECONE_API_KEY"] = PINECORE_API_KEY

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

