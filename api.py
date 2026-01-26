from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG import RAG, Context
import uvicorn

app = FastAPI(
    title="ChatEILCO API",
    description="API pour répondre aux questions sur l'EILCO en utilisant RAG",
    version="1.0.0"
)

API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "gemma"
VECTOR_DB_PATH = "./faiss_index-v2"

rag_system = RAG(api_url=API_URL, model_name=MODEL_NAME)
user_contexts = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Quels sont les horaires d'ouverture de l'EILCO?",
                "session_id": "user123"
            }
        }

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    token_count: int
    history_size: int

@app.on_event("startup")
async def startup_event():
    try:
        rag_system.load_knowledge_base(VECTOR_DB_PATH)
        print("✓ Knowledge base loaded successfully")
    except Exception as e:
        print(f"✗ Error loading knowledge base: {str(e)}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur l'API ChatEILCO",
        "endpoints": {
            "/query": "POST - Poser une question",
            "/health": "GET - Vérifier l'état de l'API",
            "/reset": "POST - Réinitialiser une session",
            "/docs": "GET - Documentation interactive"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "knowledge_base_loaded": rag_system.vectorstore is not None,
        "active_sessions": len(user_contexts)
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        if not rag_system.vectorstore:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base not loaded"
            )
        
        if request.session_id not in user_contexts:
            user_contexts[request.session_id] = Context(token_limit=4000, turns_to_leave=4)
        
        ctx = user_contexts[request.session_id]
        
        docs = rag_system.retriever(request.query)
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
        
        response = rag_system.run_turn(request.query, ctx)
        
        if response and 'choices' in response and len(response['choices']) > 0:
            answer = response['choices'][0]['message']['content']
        else:
            answer = "Désolé, je n'ai pas pu générer une réponse."
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            token_count=ctx.token_count,
            history_size=len(ctx.history)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/reset")
async def reset_session(session_id: str = "default"):
    if session_id in user_contexts:
        del user_contexts[session_id]
        return {"message": f"Session {session_id} reset successfully"}
    return {"message": f"Session {session_id} not found"}

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )