import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# Load environment variables
load_dotenv()

# Enable global LLM cache
set_llm_cache(InMemoryCache())

def get_llm():
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
