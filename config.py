from langchain_docling.loader import ExportType
from langchain_core.prompts import PromptTemplate

class Config:
    def __init__(self):
        self.dir_files = "./pdf_files/"
        self.embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.export_type = ExportType.DOC_CHUNKS
        self.question = "salah ibrahim?"
        self.prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n"
            "---------------------\nGiven the context information and not prior knowledge, "
            "answer the query.\nQuery: {question}\nAnswer:\n"
        )

        self.persist_directory = "./data/vectorstore"
        self.top_k = 3
        self.llm_name = "mistral"
        self.llm_api_url = "http://localhost:8080/v1/chat/completions"