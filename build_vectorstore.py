from OllamaStuff import Ollama_Knowledge_base,OllamaEmbeddings
import os



def build_vectorstore(input_files_path, vectordb_path, embeddings):
    
    files_paths = []
    for file_name in os.listdir(input_files_path):
        if file_name.endswith(".md"):
            file_path = os.path.join(input_files_path, file_name)
            files_paths.append(file_path)

    kb = Ollama_Knowledge_base(list_file_paths=files_paths)
    documents = kb.loader()
    chunks = kb.splitter(documents)
    return kb.storer(chunks, embeddings, vectordb_path)


if __name__ == "__main__":
    embeddings = OllamaEmbeddings("nomic-embed-text", "http://localhost:11434/api/embed")
    build_vectorstore("markdown_docs", "faiss_index", embeddings)