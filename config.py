from langchain_docling.loader import ExportType
from langchain_core.prompts import PromptTemplate
from eilco_prompts import get_rag_prompt, get_system_prompt, SCHOOL_INFO
import os

class Config:
    def __init__(self):
        self.dir_files = "./pdf_files/"
        self.embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.export_type = ExportType.DOC_CHUNKS
        self.question = "EILCO?"
        
        # Utilise le prompt personnalisé EILCO pour RAG
        self.prompt = PromptTemplate.from_template(get_rag_prompt())

        self.persist_directory = "./data/vectorstore"
        self.top_k = 6
        self.chunk_max_tokens = 1000  # Augmenté pour plus de contexte par chunk
        self.llm_name = "mistral"
        # Read LLM API URL from environment variable
        # For Docker on Linux: use your host machine IP (e.g., 192.168.1.100:8080)
        # or use 172.17.0.1:8080 (default bridge gateway)
        self.llm_api_url = os.getenv("LLM_API_URL", "http://127.0.0.1:8080/v1/chat/completions")
        
        # Configuration du rôle/persona de l'assistant
        # Options: "default", "academic", "student_support", "career", "technical_support"
        self.assistant_role = "default"
        self.system_prompt = get_system_prompt(self.assistant_role)
        
        # Informations de l'établissement
        self.school_info = SCHOOL_INFO
        
        # Query router configuration
        self.use_query_router = True