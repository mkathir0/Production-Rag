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

    search_kwargs = {"k": 10}
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

    # 1. History Aware Retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. "
        "CRITICAL INSTRUCTION: If the latest question is already standalone (e.g. 'What is SQL?'), YOU MUST RETURN IT EXACTLY AS IS. Do not add conversational filler. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # We REMOVED MultiQueryRetriever here. It was causing multiple LLM calls 
    # before retrieval, hitting Groq's 6000 TPM free tier limit and causing 
    # 2-minute retry backoffs. EnsembleRetriever alone is plenty fast and accurate!
    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )

    # 2. QA Chain
    qa_system_prompt = (
        "You are an expert assistant. You MUST answer questions using ONLY the provided context below. "
        "Do NOT use your own internal knowledge. "
        "Always start your response exactly with 'Based on your documents...\\n\\n'. "
        "If the answer is not contained in the context, say 'I cannot find the answer in the provided documents.' "
        "IMPORTANT: Format your response using Markdown! Use **bold text** for key terms, bullet points for lists, and *italics* for emphasis to make the answer highly readable."
        "\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Manual Retrieval & Streaming Generation
    # Step A: Get the documents (awaits the history reformulation + vector search)
    docs = await history_aware_retriever.ainvoke({
        "input": latest_question, 
        "chat_history": chat_history
    })
    
    # Step B: Format the prompt manually with the retrieved documents
    context_str = "\n\n".join([doc.page_content for doc in docs])
    formatted_messages = qa_prompt.format_messages(
        context=context_str,
        chat_history=chat_history,
        input=latest_question
    )
    
    # Step C: Stream directly from the LLM to get true character-by-character animation
    async for chunk in llm.astream(formatted_messages):
        if chunk.content:
            yield chunk.content
