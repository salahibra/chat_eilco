from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_core.documents import Document  
import prompts
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import requests, os, re, json
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
        if not text or not isinstance(text, str):
            return []
        
        payload = {
            "model": self.model,
            "input": text
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
        lines = text.split('\n')
        table_regions = []  
        
        in_table = False
        start_idx = -1

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

        cleaned_lines = []
        extracted_data = [] 
        
        last_processed_idx = 0
        
        for start, end in table_regions:
            cleaned_lines.extend(lines[last_processed_idx:start])
            cleaned_lines.append("\n[TABLE_SEE_METADATA_SUMMARY]\n")
            
            context_start = start
            context_end = end
            
            found_specific_context = False

            # Check Line BELOW (Priority 1)
            if end < len(lines):
                line_below = lines[end].strip()
                if re.search(r"(?i)\*?tableau\s*.*\s*:\*?", line_below):
                    context_end = end + 1 
                    found_specific_context = True

            # Check Line ABOVE (Priority 2)
            if not found_specific_context and start > 0:
                line_above = lines[start - 1].strip()
                if line_above.endswith(':'):
                    context_start = start - 1 
                    found_specific_context = True

            # FALLBACK (Priority 3)
            if not found_specific_context:
                context_start = max(0, start - 2)
                context_end = min(len(lines), end + 2)

            chunk_lines = lines[context_start:context_end]
            extracted_data.append("\n".join(chunk_lines))
            
            last_processed_idx = end

        cleaned_lines.extend(lines[last_processed_idx:])

        return "\n".join(cleaned_lines), extracted_data

    def _format_section_path(self, metadata):
        parts = []
        
        if "source" in metadata:
            filename = os.path.basename(metadata["source"])
            parts.append(filename)
        
        headers = []
        for key in sorted(metadata.keys()):
            if key.startswith("Header_"):
                headers.append(metadata[key])
        
        parts.extend(headers)
        return " > ".join(parts)

    def _summarize_table(self, table_text, section_path):
        system_prompt = prompts.SYSTEM_SUM_TABLE.format(section_path = section_path)
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Voici le tableau :\n\n{table_text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
            "cache_prompt": False,
            "model": "mistral"
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result:
                return result['choices'][0]['message']['content']
            elif 'content' in result:
                return result['content']
            else:
                return str(result)
                
        except Exception as e:
            print(f"Error summarizing table: {e}")
            return f"Résumé non disponible. (Location: {section_path})"

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
            header_splits = markdown_splitter.split_text(doc.page_content)
            
            for section_split in header_splits:
                section_split.metadata.update(doc.metadata)

                text_content, extracted_tables_with_context = self._extract_tables_with_context(section_split.page_content)
                
                section_path = self._format_section_path(section_split.metadata)

                for raw_table_context in extracted_tables_with_context:
                    
                    summary = self._summarize_table(raw_table_context, section_path)
                    
                    full_raw_content = f"**Location:** {section_path}\n\n{raw_table_context}"

                    table_doc = Document(
                        page_content=summary, 
                        metadata=section_split.metadata.copy()
                    )
                    
                    table_doc.metadata["type"] = "table_summary"
                    table_doc.metadata["original_content"] = full_raw_content 
                    
                    all_final_chunks.append(table_doc)

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
        if os.path.exists(vector_path) and os.path.exists(os.path.join(vector_path, "index.faiss")):
            self.vectorstore = FAISS.load_local(
                folder_path=vector_path, 
                embeddings=embeddings, 
                allow_dangerous_deserialization=True
            )
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = FAISS.from_documents(
                documents=chunks, 
                embedding=embeddings
            )

        self.vectorstore.save_local(vector_path)
        return self.vectorstore

    def save_to_json(self, output_path):
        if not self.chunks:
            return

        data_to_save = []
        for doc in self.chunks:
            data_to_save.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def load_from_json(self, input_path):
        if not os.path.exists(input_path):
            return

        with open(input_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        self.chunks = []
        for item in loaded_data:
            doc = Document(
                page_content=item["page_content"],
                metadata=item["metadata"]
            )
            self.chunks.append(doc)
        return self.chunks