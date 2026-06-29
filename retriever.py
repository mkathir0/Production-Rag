from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor

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
    print(f"\n--- Asking Question (Advanced RAG) ---")
    print(f"Question: {question}")
    if filter_dict:
        print(f"Filter: {filter_dict}")
    
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
    
    search_kwargs = {"k": 3}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    vector_retriever = db.as_retriever(search_kwargs=search_kwargs)
    
    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
    else:
        ensemble_retriever = vector_retriever
        
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm
    )
    
    # Due to Groq's Free Tier Token Per Minute (TPM) limits, Contextual Compression 
    # executes too many concurrent LLM calls (one per retrieved chunk), hitting the 6000 TPM limit.
    # We will pass the MultiQueryRetriever directly into the RAG chain instead.
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(multi_query_retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": question})
    return response['answer']
