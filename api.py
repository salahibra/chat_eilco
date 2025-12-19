from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG import RAG
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="ChatEILCO API",
    description="API pour répondre aux questions sur l'EILCO en utilisant RAG",
    version="1.0.0"
)

# Initialize RAG system
rag_system = RAG()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Quels sont les horaires d'ouverture de l'EILCO?"
            }
        }

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]

# Load knowledge base on startup
@app.on_event("startup")
async def startup_event():
    """Load the knowledge base when the API starts"""
    try:
        rag_system.load_knowledge_base()
        print("✓ Knowledge base loaded successfully")
    except Exception as e:
        print(f"✗ Error loading knowledge base: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bienvenue sur l'API ChatEILCO",
        "endpoints": {
            "/query": "POST - Poser une question",
            "/health": "GET - Vérifier l'état de l'API",
            "/docs": "GET - Documentation interactive"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_base_loaded": rag_system.vectorstore is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a user query using RAG system
    
    Args:
        request: QueryRequest containing the user's question
        
    Returns:
        QueryResponse with the answer and source documents
    """
    try:
        if not rag_system.vectorstore:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base not loaded"
            )
        
        # Retrieve relevant documents
        docs = rag_system.retriever(request.query)
        
        # Extract document contents and sources
        doc_contents = [doc.page_content for doc in docs]
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
        
        # Generate augmented prompt
        augmented_prompt = rag_system.prompt_augmentation(doc_contents, request.query)
        
        # Generate response
        response = rag_system.response_generator(augmented_prompt)
        
        # Extract answer from response
        if 'choices' in response and len(response['choices']) > 0:
            answer = response['choices'][0]['message']['content']
        else:
            answer = "Désolé, je n'ai pas pu générer une réponse."
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
