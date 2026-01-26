from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
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
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
            ("####", "Header_4"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False 
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", " ", ""]
        )

        all_final_chunks = []

        for doc in documents:
            titre_sections_splits = markdown_splitter.split_text(doc.page_content)
            
            for section_split in titre_sections_splits:
                section_split.metadata.update(doc.metadata)
                
            final_chunks = text_splitter.split_documents(titre_sections_splits)
            
            all_final_chunks.extend(final_chunks)

        self.chunks = all_final_chunks
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
    
