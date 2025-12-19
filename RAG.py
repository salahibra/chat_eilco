
import requests, json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
class RAG:
    def __init__(self):
        self.vectorstore = None
        self.Model_API_URL = "http://localhost:8080/v1/chat/completions"
        self.Model_NAME1 = "/home/salah/.cache/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf",
    def load_knowledge_base(self):
        self.vectorstore = FAISS.load_local("faiss_index-v2", HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'), allow_dangerous_deserialization=True)
    def retriever(self, query):
        docs = self.vectorstore.similarity_search(query, k=5)
        return docs
    def prompt_augmentation(self, docs, query):
        augmented_prompt = f"en se basant sur les documents suivants : {docs}, reponds a la question suivante : {query}"
        # À encore optimiser (Youssef)
        return augmented_prompt
    def response_generator(self, augmented_prompt):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.Model_NAME1,
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": augmented_prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "stream": False
        }
        response = requests.post(self.Model_API_URL, headers=headers, data=json.dumps(payload))
        return response.json()