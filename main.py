from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import get_llm
from database import get_or_create_vector_db, create_bm25_retriever
from retriever import ask_question

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
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer = ask_question(db, bm25_retriever, llm, request.message)
    return ChatResponse(answer=answer)
