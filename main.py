from fastapi import FastAPI, Depends, HTTPException, Security, Request, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
import secrets
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import get_llm
from database import get_or_create_vector_db, create_bm25_retriever
from retriever import ask_question, aask_question

# Standard Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Setup Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security: CORS Policy Restricting to trusted domains
origins = [
    "http://localhost:3000",
    "https://production-rag-frontend-i04qt3t99-mithrajit-kathirs-projects.vercel.app",
    "https://production-rag-frontend.vercel.app" # General vercel domain just in case
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security: API Key Authentication
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    # Clean token if Bearer prefix is used
    token = api_key.replace("Bearer ", "") if api_key.startswith("Bearer ") else api_key
    expected_key = os.environ.get("API_SECRET_KEY")
    
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server configuration error: API_SECRET_KEY not set")
        
    if not secrets.compare_digest(token, expected_key):
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Key")
    return token

# Initialize resources once at startup
from ingest import run_ingestion
run_ingestion()

llm = get_llm()
db = get_or_create_vector_db()
bm25_retriever = create_bm25_retriever(db)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

# Secure the endpoint by adding the dependency
@app.post("/api/chat", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest):
    async def stream_generator():
        async for chunk in aask_question(db, bm25_retriever, llm, body.messages):
            # For simplicity, yielding raw text tokens. 
            # If you want SSE, you would format as 'data: {chunk}\n\n'
            yield chunk
            
    return StreamingResponse(stream_generator(), media_type="text/plain")

import aiofiles

@app.post("/api/upload", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Ensure data_sources directory exists
    os.makedirs("data_sources", exist_ok=True)
    
    file_path = os.path.join("data_sources", file.filename)
    
    # Save the file asynchronously
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
        
    logger.info(f"File saved to {file_path}. Triggering ingestion...")
    
    # Trigger ingestion
    try:
        run_ingestion()
        return {"filename": file.filename, "status": "success", "message": "File successfully uploaded and ingested into the knowledge base."}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"File uploaded, but ingestion failed: {str(e)}")
