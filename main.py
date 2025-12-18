from RAG import RAG
from OllamaStuff import OllamaEmbeddings, OllamaRAG, Ollama_Knowledge_base
import os


#=============================== testing script ===========================================

if __name__ == "__main__":
    embeddings = OllamaEmbeddings("nomic-embed-text", "http://localhost:11434/api/embed")

    vectordb_path = "faiss_index"
    run_rag = True

    if not os.path.exists(vectordb_path):
        markdown_folder = "./markdown_docs/"
        files_paths = []
        for file_name in os.listdir(markdown_folder):
            if file_name.endswith(".md"):
                file_path = os.path.join(markdown_folder, file_name)
                files_paths.append(file_path)

        kb = Ollama_Knowledge_base(list_file_paths=files_paths)
        documents = kb.loader()
        chunks = kb.splitter(documents)
        vectorstore = kb.storer(chunks, embeddings, vectordb_path)

    if run_rag:
        rag = OllamaRAG(
            "http://localhost:11434/api/generate",
            "gemma3:4b",
            {"temperature": 0.5, "num_predict": 1024},
        )
        rag.load_knowledge_base(vectordb_path, embeddings)

        queries = [
            "Quelle est la durée du semestre S9 ?",
            "et pour le S8 ?",
            "quelle est la difference de durée entre les deux ?"
        ]
        for query in queries:
            docs = rag.retriever(query)
            augmented_prompt = rag.prompt_augmentation(docs, query)
            response = rag.response_generator(augmented_prompt)

            ai_content = response.get("message", {}).get("content", str(response))
            RAG.HISTORY.append({"user": query, "ai": ai_content})

            #print("\n\nHISTORY:", RAG.HISTORY)
            print(f"Response from Model:\n{ai_content}\n")