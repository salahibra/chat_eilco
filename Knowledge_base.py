from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests, json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class Knowledge_base:
    def __init__(self, list_file_paths):
        self.list_file_paths = list_file_paths
        self.api_url = "http://localhost:8080/embedding"  # Example API URL, replace with your actual URL
        self.chunks = None
        self.embeddings = None
        self.vectorstore = None
    

    def loader(self):
        data = []
        for file_path in self.list_file_paths:
            data.append(UnstructuredMarkdownLoader(file_path).load()[0])
        return data

    
    def splitter(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len
        )
        self.chunks = text_splitter.split_documents(documents)
        return self.chunks

    def converter(self, texts):
        headers = {"Content-Type": "application/json"}
        payload = {"input": [text.page_content for text in texts]}
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        self.embeddings = response.json()
        return self.embeddings
        
    def storer(self, chunks : list, path : str = "faiss_index"):
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self.vectorstore.save_local(path)
        return self.vectorstore
    