from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG import RAG
from Knowledge_base import Knowledge_base
from config import Config
from query_router import QueryRouter
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
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    # Get only the last 'limit' messages in chronological order
    history = [{"role": row[0], "content": row[1]} for row in rows[-limit:]]
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
    print(f"Received query for session {request.session_id}: {request.message}")
    hist = get_history(request.session_id)
    
    if not hist:
        hist = []
        chat_history = ""
    else:
        chat_history = "\n".join([f"{item['role']}: {item['content']}" for item in hist])
        print(f"Chat history for session {request.session_id}:\n{chat_history}")
    
    try:
        if not rag_system.retriever:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base not loaded"
            )
        
        # Use query router to classify and determine retrieval need
        if query_router:
            routing_result = query_router.route(request.message, chat_history)
            classification = routing_result["classification"]
            reasoning = routing_result["reasoning"]
            needs_retrieval = routing_result["needs_retrieval"]
            
            print(f"Query classification: {classification}")
            print(f"Reasoning: {reasoning}")
            print(f"Needs retrieval: {needs_retrieval}")
            
            # Condense query only if it's knowledge_seeking (needs context)
            if classification == "knowledge_seeking":
                condensed_query = rag_system.condense_query(chat_history, request.message)
                print(f"Condensed query: {condensed_query}")
            else:
                # For conversational/ambiguous, use the original message
                condensed_query = request.message
                print(f"No condensing needed for {classification} query")
        else:
            # Fallback to old behavior if router is disabled
            print("Query router disabled, using legacy condensing")
            condensed_query = rag_system.condense_query(chat_history, request.message)
            print(f"Condensed query: {condensed_query}")
            needs_retrieval = True
        
        # Retrieve documents if needed
        if needs_retrieval:
            docs = rag_system.retriever.invoke(condensed_query)
            print(f"Retrieved {len(docs)} relevant documents for query: {condensed_query}")
            for i, doc in enumerate(docs):
                print(f"Document {i+1} content length: {len(doc.page_content)}")
            sources = rag_system.sources_as_list(docs)
            augmented_prompt = rag_system.augment_prompt(condensed_query, docs)
        else:
            print(f"Query doesn't need retrieval: {condensed_query}")
            docs = []
            sources = []
            augmented_prompt = rag_system.prompt.format(context="", question=condensed_query)
        
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
        port=8072,
        reload=True
    )
