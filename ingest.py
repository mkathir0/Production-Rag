import glob
import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter
)
from database import get_or_create_vector_db

def load_file(file_path: str):
    print(f"Loading {file_path}...")
    loader = UnstructuredFileLoader(file_path)
    return loader.load()

def populate_vector_db(db, documents):
    print("\n--- Chunking Documents ---")
    
    char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    token_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=50, allowed_special="all")
    md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get("source", "").lower()
        
        # Route the document to the appropriate splitter based on its file extension
        if source.endswith(".md"):
            chunks = md_splitter.split_documents([doc])
        elif source.endswith(".txt") or source.endswith(".csv"):
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

def run_ingestion():
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
    # Only scanning data_sources now
    all_files.extend(glob.glob("data_sources/**/*.*", recursive=True))
    
    # 3. Filter for new files only
    new_files = [f for f in all_files if f not in existing_sources and os.path.isfile(f)]
    
    if new_files:
        print(f"Found {len(new_files)} new document(s) to ingest.")
        new_documents = []
        for file in new_files:
            try:
                new_documents.extend(load_file(file))
            except Exception as e:
                print(f"Failed to load {file}: {e}")
                
        if new_documents:
            populate_vector_db(db, new_documents)
    else:
        print(f"Database already up-to-date with {db._collection.count()} chunks. Skipping chunking phase.")

if __name__ == "__main__":
    run_ingestion()
