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
    # Keeping the original function just in case
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
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(multi_query_retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": question})
    return response['answer']

from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

async def aask_question(db, bm25_retriever, llm, messages: list, filter_dict: dict = None):
    print(f"\n--- Asking Question Async (Streaming & History) ---")
    
    # Extract the latest user question
    if not messages:
        yield "No question provided."
        return
        
    latest_question = messages[-1].get("content", "")
    
    # Convert history dicts to Langchain Message objects (excluding the latest question)
    chat_history = []
    for msg in messages[:-1]:
        if msg.get("role") == "user":
            chat_history.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            chat_history.append(AIMessage(content=msg.get("content", "")))

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

    # 1. History Aware Retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, multi_query_retriever, contextualize_q_prompt
    )

    # 2. QA Chain
    qa_system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer based on the context, "
        "say that you don't know. Keep your answer clear and concise."
        "\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. RAG Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Stream the output
    async for event in rag_chain.astream_events(
        {"input": latest_question, "chat_history": chat_history}, 
        version="v1"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # We only want to yield tokens from the final QA chain, not from the retriever rewriting the question
            # The QA chain's LLM usually doesn't have a specific tag unless set, but we can filter by the node
            # The event["name"] for ChatGroq might be ChatGroq. Let's yield all model stream for now, 
            # wait, that might yield the reformulated query as well.
            # To be safe, we check if the event is from the stuff_documents_chain or similar.
            pass
            
    # Actually, astream_events can be complex to filter. 
    # An easier way is just picking the `answer` from astream on the final chain.
    async for chunk in rag_chain.astream({"input": latest_question, "chat_history": chat_history}):
        if "answer" in chunk:
            yield chunk["answer"]
