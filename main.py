import os
import shutil
import glob
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter
)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv()

def load_text_file(file_path: str):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def load_pdf_file(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print("-" * 40)
    return documents

def get_or_create_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize the database with a collection name
    db = Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings, 
        persist_directory="./chroma_db"
    )
    return db

def populate_vector_db(db, documents):
    print("\n--- Chunking Documents ---")
    
    char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    token_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=50) # Tokens contain multiple characters
    md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get("source", "").lower()
        
        # Route the document to the appropriate splitter based on its file extension
        if source.endswith(".md"):
            chunks = md_splitter.split_documents([doc])
        elif source.endswith(".txt"):
            chunks = char_splitter.split_documents([doc])
        elif source.endswith(".pdf"):
            chunks = token_splitter.split_documents([doc])
        else:
            chunks = recursive_splitter.split_documents([doc])
            print(f"Used RecursiveCharacterTextSplitter for {source}")
            
        all_chunks.extend(chunks)

    print(f"Split {len(documents)} original pages/files into {len(all_chunks)} chunks.")
    
    print("\n--- Adding to ChromaDB ---")
    db.add_documents(all_chunks)
    print("Vector database populated successfully!")

def create_bm25_retriever(db):
    print("\n--- Building BM25 Index ---")
    db_data = db.get()
    
    docs = []
    for doc_text, metadata in zip(db_data['documents'], db_data['metadatas']):
        docs.append(Document(page_content=doc_text, metadata=metadata))
        
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    print("BM25 index built successfully from Chroma cache!")
    return bm25_retriever

def perform_similarity_search(db, query: str, filter_dict: dict = None):
    print(f"\n--- Performing Similarity Search (with scores) ---")
    print(f"Query: '{query}'")
    if filter_dict:
        print(f"Filter: {filter_dict}")
    
    # ChromaDB returns distance scores (lower is better/more similar)
    results = db.similarity_search_with_score(query, k=3, filter=filter_dict)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[Match {i}] (Distance: {score:.4f} | Source: {doc.metadata.get('source')} | Page {doc.metadata.get('page', 'Unknown')}):")
        print(f"{doc.page_content.strip()}")
    print("-" * 40)

def ask_question(db, bm25_retriever, llm, question: str, filter_dict: dict = None):
    print(f"\n--- Asking Question (Hybrid Search) ---")
    print(f"Question: {question}")
    if filter_dict:
        print(f"Filter: {filter_dict}")
    
    # 1. Define how the LLM should behave and where to put the retrieved context
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer based on the context, "
        "say that you don't know. Keep your answer clear and concise."
        "\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 2. Set up the Database Retriever (fetch top 3 most relevant chunks)
    search_kwargs = {"k": 3}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    vector_retriever = db.as_retriever(search_kwargs=search_kwargs)
    
    # Initialize Ensemble Retriever combining BM25 and Vector Search
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    # 3. Combine the Retriever, the Prompt, and the LLM into a RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)
    
    # 4. Run the chain!
    response = rag_chain.invoke({"input": question})
    print(f"\nAnswer: {response['answer']}")


def main():
    # Initialize the LLM
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    print("Initializing Database...")
    db = get_or_create_vector_db()
    
    # 1. Get existing sources from ChromaDB
    existing_sources = set()
    db_data = db.get(include=["metadatas"])
    for meta in db_data.get('metadatas', []):
        if meta and 'source' in meta:
            existing_sources.add(meta['source'])
            
    # 2. Scan directories for available files
    all_files = []
    all_files.extend(glob.glob("txt/**/*.txt", recursive=True))
    all_files.extend(glob.glob("pdfs/**/*.pdf", recursive=True))
    
    # 3. Filter for new files only
    new_files = [f for f in all_files if f not in existing_sources]
    
    if new_files:
        print(f"Found {len(new_files)} new document(s) to ingest.")
        new_documents = []
        for file in new_files:
            try:
                if file.endswith('.pdf'):
                    new_documents.extend(load_pdf_file(file))
                elif file.endswith('.txt'):
                    new_documents.extend(load_text_file(file))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
                
        if new_documents:
            populate_vector_db(db, new_documents)
    else:
        print(f"Database already up-to-date with {db._collection.count()} chunks. Skipping chunking phase.")
        
    bm25_retriever = create_bm25_retriever(db)
        
    # Interactive Chat Loop!
    print("\n=== RAG System Ready ===")
    print("Type a question to get an AI answer.")
    print("Type '/search <your query>' to perform a raw similarity search.")
    print("Type '/filter <filename>' to filter by source (e.g. '/filter txt/sample.txt').")
    print("Type '/filter clear' to remove all filters.")
    print("Type 'quit' to exit.")
    
    current_filter = None
    
    while True:
        user_input = input(f"\n[Filter: {current_filter}] Input: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input.strip() == "":
            continue
            
        if user_input.startswith("/filter "):
            filter_val = user_input[8:].strip()
            if filter_val.lower() == "clear":
                current_filter = None
                print("Filter cleared.")
            else:
                current_filter = {"source": filter_val}
                print(f"Filter set to: {current_filter}")
            continue
            
        if user_input.startswith("/search "):
            search_query = user_input[8:].strip()
            perform_similarity_search(db, search_query, current_filter)
        else:
            ask_question(db, bm25_retriever, llm, user_input, current_filter)

if __name__ == "__main__":
    main()
