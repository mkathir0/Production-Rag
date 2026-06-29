from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

def get_or_create_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize the database with a collection name
    db = Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings, 
        persist_directory="./chroma_db"
    )
    return db

def create_bm25_retriever(db):
    print("\n--- Building BM25 Index ---")
    db_data = db.get()
    
    docs = []
    for doc_text, metadata in zip(db_data['documents'], db_data['metadatas']):
        docs.append(Document(page_content=doc_text, metadata=metadata))
        
    if not docs:
        return None
        
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    print("BM25 index built successfully from Chroma cache!")
    return bm25_retriever
