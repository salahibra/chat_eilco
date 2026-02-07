from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG import RAG
from Knowledge_base import Knowledge_base
from config import Config
from query_router import QueryRouter
import uvicorn
import sqlite3
import os


##############################################################################
DB = "chat_eilco.db"

def init_db():
    """Initialiser la base de données avec la table des messages si elle n'existe pas."""
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

class Chat(BaseModel):
    session_id: str
    message: str

def save_message(session_id: str, role: str, content: str):
    """Enregistrer un message dans la base de données avec le session_id, le rôle et le contenu donnés."""
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()
def get_history(session_id: str, limit: int = 6):
    """Récupérer les derniers messages 'limit' pour un session_id donné depuis la base de données."""
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    # Get only the last 'limit' messages in chronological order
    history = []
    for row in rows[-limit:]:
        role, content = row[0], row[1]
        # Clip assistant content to first 100 characters
        if role == "assistant" and len(content) > 50:
            content = content[:50] + "..."
        history.append({"role": role, "content": content})
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
    persist_directory=config.persist_directory,
    chunk_max_tokens=config.chunk_max_tokens
)
rag_system = RAG(
    model_api_url=config.llm_api_url, 
    model_name=config.llm_name,
    retriever=kb.retriever,
    prompt=config.prompt
)

# Initialize query router if enabled
if config.use_query_router:
    query_router = QueryRouter(
        llm_api_url=config.llm_api_url,
        llm_name=config.llm_name
    )
else:
    query_router = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]

@app.on_event("startup")
async def startup_event():
    """Initialiser la base de données et vérifier si la base de connaissances est chargée au démarrage"""
    init_db()
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
    Répondre à une question utilisateur en utilisant le système RAG
    
    Args:
        request: Chat contenant l'ID de session de l'utilisateur et le message
        
    Returns:
        QueryResponse avec la réponse et les documents sources
    """
    print(f"Requête reçue pour la session {request.session_id} : {request.message}")
    hist = get_history(request.session_id)
    
    if not hist:
        hist = []
        chat_history = ""
    else:
        chat_history = "\n".join([f"{item['role']}: {item['content']}" for item in hist])
        print(f"Historique de chat pour la session {request.session_id}:\n{chat_history}")
    
    try:
        if not rag_system.retriever:
            raise HTTPException(
                status_code=503,
                detail="Base de connaissances non chargée"
            )
        
        # Utiliser le routeur de requêtes pour classifier et déterminer le besoin de récupération
        if query_router:
            routing_result = query_router.route(request.message, chat_history)
            classification = routing_result["classification"]
            reasoning = routing_result["reasoning"]
            needs_retrieval = routing_result["needs_retrieval"]
            
            print(f"Query classification: {classification}")
            print(f"Reasoning: {reasoning}")
            print(f"Needs retrieval: {needs_retrieval}")
        else:
            # Fallback to old behavior if router is disabled
            print("Query router disabled")
            needs_retrieval = True
        
        # Condense query before retriever for better document matching
        condensed_query = rag_system.condense_query_for_retriever(request.message, chat_history=hist)
        print(f"Requête condensée pour le récupérateur : {condensed_query}")
        
        # Retrieve documents if needed
        if needs_retrieval:
            docs = rag_system.retriever.invoke(condensed_query)
            print(f"Récupéré {len(docs)} documents pertinents pour la requête : {condensed_query}")
            for i, doc in enumerate(docs):
                print(f"Longueur du contenu du document {i+1} : {len(doc.page_content)}")
            sources = rag_system.sources_as_list(docs)
            augmented_prompt = rag_system.augment_prompt(request.message, docs)
        else:
            print(f"La requête n'a pas besoin de récupération : {request.message}")
            docs = []
            sources = []
            augmented_prompt = rag_system.prompt.format(context="", question=request.message)
        
        response = rag_system.response_generator(augmented_prompt, chat_history=hist)
        
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
            detail=f"Erreur lors du traitement de la requête : {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=config.port if config.port else config.default_port,
        reload=True
    )
