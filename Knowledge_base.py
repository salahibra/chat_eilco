from langchain_docling.loader import ExportType
from langchain_community.document_loaders import DirectoryLoader
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from context_merger import create_enhanced_retriever

import requests, json
import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings

class Knowledge_base:
    def __init__(self, dir_files: str, export_type=ExportType.DOC_CHUNKS, embedding_model_id: str='sentence-transformers/all-MiniLM-L6-v2', persist_directory: str="./data/vectorstore", top_k: int=5, chunk_max_tokens: int=500):
        self.dir_files = dir_files
        self.chunks = None
        self.EXPORT_TYPE = export_type
        self.EMBED_MODEL_ID = embedding_model_id
        self.vectorstore = None
        self.persist_directory = persist_directory
        self.chunk_max_tokens = chunk_max_tokens  # Configure chunk size
        self.top_k = top_k
        self.load_vectorstore()
        # Use enhanced retriever with context window expansion
        # Retrieves top-k results and includes surrounding chunks (before/after)
        self.retriever = create_enhanced_retriever(
            self.vectorstore, 
            top_k=top_k,
            context_window=1  # Include 1 chunk before and 1 after each result
        )
    

    def loader(self):
        loader = DirectoryLoader(
            self.dir_files,
            glob="**/*.pdf",
            loader_cls=DoclingLoader,
            loader_kwargs={
                "chunker": HybridChunker(max_tokens=self.chunk_max_tokens),
                "export_type": self.EXPORT_TYPE,
            },
        )
            
        documents = loader.load()
        return documents
    

    def splitter(self, documents):
        """"
        Split documents into chunks based on the configured export type.
        For DOC_CHUNKS, use the pre-chunked documents from Docling.
        For MARKDOWN, further split using markdown headers as separators.
        Args:
            documents: List of documents to split into chunks
        """
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
        """Ingest the chunks into a FAISS vector store with HuggingFace embeddings.
        Each chunk is embedded and added to the vector store for retrieval.
        Returns:
            The FAISS vector store instance after ingestion
        """
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
        """Load the FAISS vector store from disk if it exists, otherwise perform loading, splitting, and ingestion.
        This method checks if a persisted vector store exists. If it does, it loads it directly. If not, it goes through the full process of loading documents, splitting them into chunks, and ingesting them into a new vector store, which is then saved for future use.
        """
        if os.path.exists(self.persist_directory):
            self.vectorstore = FAISS.load_local(self.persist_directory, HuggingFaceEmbeddings(model_name=self.EMBED_MODEL_ID), allow_dangerous_deserialization=True)
        else:
            documents = self.loader()
            self.splitter(documents)
            self.vectorstore = self.ingestion()