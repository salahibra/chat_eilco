from langchain_docling.loader import ExportType
from langchain_core.prompts import PromptTemplate
from eilco_prompts import get_rag_prompt, get_system_prompt, SCHOOL_INFO

class Config:
    def __init__(self):
        self.dir_files = "./pdf_files/"
        self.embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.export_type = ExportType.DOC_CHUNKS
        self.question = "EILCO?"
        
        # Utilise le prompt personnalisé EILCO pour RAG
        self.prompt = PromptTemplate.from_template(get_rag_prompt())

        self.persist_directory = "./data/vectorstore"
        self.top_k = 5
        self.chunk_max_tokens = 1500  # Augmenté pour plus de contexte par chunk
        self.llm_name = "mistral"
        self.llm_api_url = "http://localhost:8080/v1/chat/completions"
        
        # Configuration du rôle/persona de l'assistant
        # Options: "default", "academic", "student_support", "career", "technical_support"
        self.assistant_role = "default"
        self.system_prompt = get_system_prompt(self.assistant_role)
        
        # Informations de l'établissement
        self.school_info = SCHOOL_INFO
        
        # Query router configuration
        self.use_query_router = True