import os
import shutil
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_text_file(file_path: str):
    print(f"\n--- Loading Text File: {file_path} ---")
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def load_pdf_file(file_path: str):
    print(f"\n--- Loading PDF File: {file_path} ---")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print("-" * 40)
    return documents

def create_and_print_vector_db(documents):
    print("\n--- Chunking Documents ---")
    # Here is where we define the Recursive Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} original pages/files into {len(chunks)} chunks.")

    print("\n--- Creating ChromaDB Vector Store ---")
    # Delete the old database so we don't get duplicates when running multiple times
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # We use a fast, free local embedding model from HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist the database with the CHUNKS, not the raw documents
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    print("Vector database created and embedded successfully!")
    print(f"Total document chunks in database: {db._collection.count()}")
    return db

def ask_question(db, llm, question: str):
    print(f"\n--- Asking Question ---")
    print(f"Question: {question}")
    
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
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 3. Combine the Retriever, the Prompt, and the LLM into a RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 4. Run the chain!
    response = rag_chain.invoke({"input": question})
    print(f"\nAnswer: {response['answer']}")
    print("-" * 40)


def main():
    print("Initializing RAG System...")
    # Initialize the LLM
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    all_documents = []
    
    # Load Documents
    try:
        all_documents.extend(load_text_file("txt/sample.txt"))
    except Exception as e:
        print(f"Could not load txt/sample.txt: {e}")
        
    try:
        all_documents.extend(load_pdf_file("pdfs/iso27001.pdf"))
    except Exception as e:
        print(f"Could not load pdfs/iso27001.pdf: {e}")
        
    # Create DB and Ask Question
    if all_documents:
        db = create_and_print_vector_db(all_documents)
        
        # Interactive Chat Loop!
        print("\n=== RAG System Ready ===")
        while True:
            user_question = input("\nAsk a question (or type 'quit' to exit): ")
            if user_question.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_question.strip() == "":
                continue
                
            ask_question(db, llm, user_question)
    else:
        print("\nNo documents were loaded, skipping database creation.")

if __name__ == "__main__":
    main()
