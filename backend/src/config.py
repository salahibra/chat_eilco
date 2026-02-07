from langchain_docling.loader import ExportType
from langchain_core.prompts import PromptTemplate
from eilco_prompts import get_rag_prompt, get_system_prompt, SCHOOL_INFO
import os
from pathlib import Path

class Config:
    def __init__(self):
        # Define paths relative to this config file, not current working directory
        config_dir = Path(__file__).resolve().parent
        backend_dir = config_dir.parent
        
        self.dir_files = str(backend_dir / "data" / "pdf_files")
        self.embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.export_type = ExportType.DOC_CHUNKS
        self.question = "EILCO?"
        
        # Utilise le prompt personnalisé EILCO pour RAG
        self.prompt = PromptTemplate.from_template(get_rag_prompt())

        self.persist_directory = str(backend_dir / "data" / "vectorstore")
        self.top_k = 6
        self.chunk_max_tokens = 1000  # Augmenté pour plus de contexte par chunk
        self.llm_name = "mistral"
        # Lire l'URL de l'API LLM depuis la variable d'environnement

        self.llm_api_url = os.getenv("LLM_API_URL", "http://127.0.0.1:8080/v1/chat/completions")
        
        # Configuration du rôle/persona de l'assistant
        # Options: "default", "academic", "student_support", "career", "technical_support"
        self.assistant_role = "default"
        self.system_prompt = get_system_prompt(self.assistant_role)
        
        # Informations de l'établissement
        self.school_info = SCHOOL_INFO
        
        # Configuration du routeur de requêtes
        self.use_query_router = True
        # FastAPI numero de port par défaut
        self.port = 8000
        # FastAPI numero de port si le port par défaut est utilisé
        self.default_port = 8072