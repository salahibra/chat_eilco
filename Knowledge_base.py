from langchain_community.document_loaders import UnstructuredMarkdownLoader,TextLoader
from langchain_core.documents import Document  
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import requests, os, re
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class LLamaCppEmbeddings(Embeddings):
    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        payload = {
            "model": self.model,
            "input": text,
            "cache_prompt": False
        }
        r = requests.post(self.url, json=payload)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]


class Knowledge_base:
    def __init__(self, list_file_paths, api_url):
        self.list_file_paths = list_file_paths
        self.api_url = api_url
        self.chunks = None
        self.embeddings = None
        self.vectorstore = None
    
    def loader(self):
        data = []
        for file_path in self.list_file_paths:
            data.append(TextLoader(file_path, encoding='utf-8').load()[0])
        return data

    def _extract_tables_with_context(self, text):
        """
        Detects markdown tables, extracts them WITH 2 lines of context above/below,
        and removes the table from the original text.
        """
        lines = text.split('\n')
        table_regions = []  
        
        in_table = False
        start_idx = -1

        # 1. Detect Tables
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                if not in_table:
                    in_table = True
                    start_idx = i
            else:
                if in_table:
                    in_table = False
                    end_idx = i 
                    table_regions.append((start_idx, end_idx))
        
        if in_table:
            table_regions.append((start_idx, len(lines)))

        # 2. Extract and Clean
        cleaned_lines = []
        extracted_data = [] 
        
        last_processed_idx = 0
        
        for start, end in table_regions:
            cleaned_lines.extend(lines[last_processed_idx:start])
            cleaned_lines.append("\n[TABLE_SEE_METADATA_SUMMARY]\n")
            
            # Get 2 lines context
            context_start = max(0, start - 2)
            context_end = min(len(lines), end + 2)
            
            chunk_lines = lines[context_start:context_end]
            extracted_data.append("\n".join(chunk_lines))
            
            last_processed_idx = end

        cleaned_lines.extend(lines[last_processed_idx:])

        return "\n".join(cleaned_lines), extracted_data

    def _format_section_path(self, metadata):
        """
        Helper: reconstructing the header path from metadata 
        (e.g., 'Chapter 1 > Section 2.1')
        """
        headers = []
        # MarkdownHeaderTextSplitter uses keys like Header_1, Header_2, etc.
        for key in sorted(metadata.keys()):
            if key.startswith("Header_"):
                headers.append(metadata[key])
        return " > ".join(headers)

    def _summarize_table(self, table_text, section_path):
        """
        Sends the table text + section context to the Llama server.
        """
        # Updated prompt to include section context
        system_prompt = (
            "Tu es un assistant expert en analyse de données. "
            "Ta tâche est de résumer le tableau Markdown suivant. "
            f"Le tableau se trouve dans la section : '{section_path}'. "
            "Le tableau est fourni avec les lignes de contexte environnantes. "
            "Donne un résumé descriptif et concis en Français qui capture les données clés, les mots clés "
            "les légendes et les tendances, en prenant en compte la section où il se trouve." \
            "ne dépasse pas 2000 caractères."
        )
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Voici le tableau :\n\n{table_text}"}
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "cache_prompt": False,
            "model":"mistral"
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Adjust parsing logic for your specific Llama server format
            if 'choices' in result:
                return result['choices'][0]['message']['content']
            elif 'content' in result:
                return result['content']
            else:
                return str(result)
                
        except Exception as e:
            print(f"Error summarizing table: {e}")
            return f"Résumé non disponible. (Section: {section_path})"

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
        summaries = []

        for doc in documents:
            # 1. Split by Headers first
            header_splits = markdown_splitter.split_text(doc.page_content)
            
            for section_split in header_splits:
                # Merge original file metadata
                section_split.metadata.update(doc.metadata)

                # 2. Extract Tables
                text_content, extracted_tables_with_context = self._extract_tables_with_context(section_split.page_content)
                
                section_path = self._format_section_path(section_split.metadata)

                # A. Process Tables
                for raw_table_context in extracted_tables_with_context:
                    
                    # Pass section path to summarizer
                    summary = self._summarize_table(raw_table_context, section_path)
                    summaries.append(summary)
                    # Prepare the raw content string (Headers + Table) for storage
                    full_raw_content = f"**Section Context:** {section_path}\n\n{raw_table_context}"

                    # Create Document: Content = Summary
                    table_doc = Document(
                        page_content=summary, 
                        metadata=section_split.metadata.copy()
                    )
                    
                    # Store Raw Table + Context in metadata
                    table_doc.metadata["type"] = "table_summary"
                    table_doc.metadata["original_content"] = full_raw_content 
                    
                    all_final_chunks.append(table_doc)

                # B. Process Text
                clean_doc = Document(
                    page_content=text_content,
                    metadata=section_split.metadata.copy()
                )
                clean_doc.metadata["type"] = "text"
                
                text_chunks = text_splitter.split_documents([clean_doc])
                all_final_chunks.extend(text_chunks)

        self.chunks = all_final_chunks
        return self.chunks
    
 
    def storer(self, chunks, embeddings: Embeddings, vector_path: str):
        """
        Checks if a FAISS vectorstore exists at vector_path.
        - If YES: Loads it and appends new chunks.
        - If NO: Creates a new one from chunks.
        Finally, saves the updated index to disk.
        """
        
        # Check if the path and the specific index file exist
        if os.path.exists(vector_path) and os.path.exists(os.path.join(vector_path, "index.faiss")):
            print(f"Found existing vectorstore at {vector_path}. Loading...")
            
            # Load the existing vectorstore
            # Note: allow_dangerous_deserialization is required in newer LangChain versions
            # Only set this to True if you trust the file source (which is your own local disk).
            self.vectorstore = FAISS.load_local(
                folder_path=vector_path, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Append the new documents to the existing index
            self.vectorstore.add_documents(chunks)
            print(f"Appended {len(chunks)} new documents to the existing store.")
            
        else:
            print(f"No existing store found at {vector_path}. Creating new one...")
            
            # Create a fresh vectorstore
            self.vectorstore = FAISS.from_documents(
                documents=chunks, 
                embedding=embeddings
            )
            print(f"Created new store with {len(chunks)} documents.")

        # Save the updated (or new) index back to disk
        self.vectorstore.save_local(vector_path)
        print(f"Vectorstore saved successfully to {vector_path}")
        
        return self.vectorstore