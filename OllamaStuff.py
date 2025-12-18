import requests
from langchain.embeddings.base import Embeddings
from RAG import RAG
from langchain_community.vectorstores import FAISS
from Knowledge_base import Knowledge_base


class Ollama_Knowledge_base(Knowledge_base):
    def __init__(self, list_file_paths, model = None, api_url = None):
        super().__init__(list_file_paths)
    
    def storer(self, chunks: list, embeddings : Embeddings, path: str = "faiss_index"):
        embeddings = embeddings
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self.vectorstore.save_local(path)
        return self.vectorstore


class OllamaRAG(RAG):
    def __init__(self, api_url, model_name, options : dict[str:int]):
        super().__init__(api_url, model_name)
        self.options = options
    
    def load_knowledge_base(self, path: str, embeddings : Embeddings):
        self.vectorstore = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def response_generator(self, augmented_prompt):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.Model_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": augmented_prompt},
            ],
            "stream": False,
            "options": self.options,
        }

        response = requests.post(
            "http://localhost:11434/api/chat", json=payload, headers=headers
        )
        return response.json()

class OllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model,
        url
    ):
        self.model = model
        self.url = url

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        payload = {
            "model": self.model,
            "input": text
        }
        r = requests.post(self.url, json=payload)
        r.raise_for_status()
        return r.json()["embeddings"][0]
