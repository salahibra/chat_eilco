from langchain_docling.loader import ExportType
from langchain_community.document_loaders import DirectoryLoader
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker


import requests, json
import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings

class Knowledge_base:
    def __init__(self, dir_files: str, export_type=ExportType.DOC_CHUNKS, embedding_model_id: str='sentence-transformers/all-MiniLM-L6-v2', persist_directory: str="./data/vectorstore", top_k: int=5):
        self.dir_files = dir_files
        self.chunks = None
        self.EXPORT_TYPE = export_type
        self.EMBED_MODEL_ID = embedding_model_id
        self.vectorstore = None
        self.persist_directory = persist_directory
        self.load_vectorstore()
        self.retriever =  self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
    

    def loader(self):
        loader = DirectoryLoader(
            self.dir_files,
            glob="**/*.pdf",
            loader_cls=DoclingLoader,
            loader_kwargs={
                "chunker": HybridChunker(tokenizer=self.EMBED_MODEL_ID),
                "export_type": self.EXPORT_TYPE,
            },
        )
            
        documents = loader.load()
        return documents
    

    def splitter(self, documents):
        if self.EXPORT_TYPE == ExportType.DOC_CHUNKS:
            self.chunks = documents
        elif self.EXPORT_TYPE == ExportType.MARKDOWN:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header_1"),
                    ("##", "Header_2"),
                    ("###", "Header_3"),
                ],
            )
            self.chunks = [split for doc in documents for split in splitter.split_text(doc.page_content)]
        else:
            raise ValueError(f"Unexpected export type: {self.EXPORT_TYPE}")
        
    def ingestion(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.EMBED_MODEL_ID)
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        self.vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        from uuid import uuid4
        uuids = [str(uuid4()) for _ in range(len(self.chunks))]
        self.vectorstore.add_documents(documents=self.chunks, ids=uuids)
        self.vectorstore.save_local(self.persist_directory)
        return self.vectorstore
    def load_vectorstore(self):
        if os.path.exists(self.persist_directory):
            self.vectorstore = FAISS.load_local(self.persist_directory, HuggingFaceEmbeddings(model_name=self.EMBED_MODEL_ID), allow_dangerous_deserialization=True)
        else:
            documents = self.loader()
            self.splitter(documents)
            self.vectorstore = self.ingestion()