from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from config import get_llm
from database import get_or_create_vector_db, create_bm25_retriever
from retriever import ask_question, aask_question

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize resources once at startup
llm = get_llm()
db = get_or_create_vector_db()
bm25_retriever = create_bm25_retriever(db)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    async def stream_generator():
        async for chunk in aask_question(db, bm25_retriever, llm, request.messages):
            # For simplicity, yielding raw text tokens. 
            # If you want SSE, you would format as 'data: {chunk}\n\n'
            yield chunk
            
    return StreamingResponse(stream_generator(), media_type="text/plain")
