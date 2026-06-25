from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader

load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
response = llm.invoke("Hello, how are you?")
print(response.content)


def load_text_file(file_path: str):
    print(f"\n--- Loading Text File: {file_path} ---")
    loader = TextLoader(file_path)
    documents = loader.load()
    for doc in documents:
        print(doc.page_content)
    print("-" * 40)
    return documents

def load_pdf_file(file_path: str):
    print(f"\n--- Loading PDF File: {file_path} ---")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        print(f"Page {doc.metadata.get('page', 'Unknown')}:")
        print(doc.page_content)
    print("-" * 40)
    return documents

def main():
    print("Hello from rag!")
    # Example usage:
    load_text_file("txt/sample.txt")
    load_pdf_file("pdfs/iso27001.pdf")

if __name__ == "__main__":
    main()
