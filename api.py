from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG import RAG
from Knowledge_base import Knowledge_base
from config import Config
import uvicorn
import sqlite3


##############################################################################
DB = "chat_eilco.db"
class Chat(BaseModel):
    session_id: str
    message: str
def save_message(session_id: str, role: str, content: str):
    """Save a message to the database with the given session_id, role, and content."""
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()
def get_history(session_id: str, limit: int = 6):
    """Retrieve the last 'limit' messages for a given session_id from the database."""
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    history = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    return history

###############################################################################


app = FastAPI(
    title="ChatEILCO API",
    description="API pour répondre aux questions sur l'EILCO en utilisant RAG",
    version="1.0.0"
)
config = Config()
kb = Knowledge_base(
    dir_files=config.dir_files, 
    export_type=config.export_type, 
    embedding_model_id=config.embedding_model_id, 
    top_k=config.top_k,
    persist_directory=config.persist_directory
)
rag_system = RAG(
    model_api_url=config.llm_api_url, 
    model_name=config.llm_name,
    retriever=kb.retriever,
    prompt=config.prompt
)


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]

@app.on_event("startup")
async def startup_event():
    """check if knowledge base is loaded on startup"""
    if rag_system.retriever is None:
        raise RuntimeError("Knowledge base not loaded. Please build the knowledge base before starting the API.")
    
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bienvenue sur l'API ChatEILCO",
        "endpoints": {
            "/chat": "POST - Poser une question",
            "/health": "GET - Vérifier l'état de l'API",
            "/docs": "GET - Documentation interactive"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_base_loaded": rag_system.retriever is not None
    }

@app.post("/chat", response_model=QueryResponse)
async def chat(request: Chat):
    """
    Answer a user query using RAG system
    
    Args:
        request: Chat containing the user's session ID and message
        
    Returns:
        QueryResponse with the answer and source documents
    """
    hist = get_history(request.session_id)
    if not hist:
        hist = []
        chat_history = ""
        condensed_query = request.message
    else:
        chat_history = "\n".join([f"{item['role']}: {item['content']}" for item in hist])
        condensed_query = rag_system.condense_query(chat_history, request.message)
    try:
        if not rag_system.retriever:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base not loaded"
            )
        
        docs = rag_system.retriever.invoke(condensed_query)
        
        sources = rag_system.sources_as_list(docs)
        
        augmented_prompt = rag_system.augment_prompt(condensed_query, docs)
        
        response = rag_system.response_generator(augmented_prompt)
        
        if 'choices' in response and len(response['choices']) > 0:
            answer = response['choices'][0]['message']['content']
            save_message(request.session_id, "user", request.message)
            save_message(request.session_id, "assistant", answer)

        else:
            answer = "Désolé, je n'ai pas pu générer une réponse."
        
        return QueryResponse(
            query=request.message,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
