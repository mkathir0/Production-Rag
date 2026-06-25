from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
response = llm.invoke("Hello, how are you?")
print(response.content)


def main():
    print("Hello from rag!")


if __name__ == "__main__":
    main()
