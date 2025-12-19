from Knowledge_base import Knowledge_base
from RAG import RAG
import os

build_knowledge_base = True
run_rag = True

if build_knowledge_base:
    markdown_folder = "./markdown_docs/"
    files_paths = []
    for file_name in os.listdir(markdown_folder):
        if file_name.endswith(".md"):
            file_path = os.path.join(markdown_folder, file_name)
            files_paths.append(file_path)
    
    kb = Knowledge_base(list_file_paths=files_paths)
    documents = kb.loader()
    chunks  = kb.splitter(documents)
    vectorstore = kb.storer(chunks)

if run_rag:
    rag = RAG()
    rag.load_knowledge_base()
    query = "quels sont les prérequis, les objectifs et le programme pour le module Microbiologie générale?"
    docs = rag.retriever(query)
    augmented_prompt = rag.prompt_augmentation(docs, query)
    response = rag.response_generator(augmented_prompt)
    print(f"Response from Model: \n {response['choices'][0]['message']['content']}")