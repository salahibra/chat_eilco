import os
from RAG import RAG, Context
from Knowledge_base import LLamaCppEmbeddings, Knowledge_base
from langchain_community.embeddings import HuggingFaceEmbeddings

API_URL = "http://localhost:8000/v1/chat/completions"
EMB_URL = "http://localhost:8080/v1/embeddings"
EMB_MODEL_NAME = "jina"
MODEL_NAME = "mistral"

#emb = LLamaCppEmbeddings(EMB_MODEL_NAME, EMB_URL)
emb2 = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

VECTOR_DB_PATH = "./faiss_index-v3"
EMB = emb2

build_knowledge_base = True

if build_knowledge_base:
    markdown_folder = "./markdown_docs/"
    files_paths = []
    for file_name in os.listdir(markdown_folder):
        if file_name.endswith(".md"):
            file_path = os.path.join(markdown_folder, file_name)
            files_paths.append(file_path)
    
    kb = Knowledge_base(list_file_paths=files_paths,api_url=API_URL)
    documents = kb.loader()
    chunks  = kb.splitter(documents)
    vectorstore = kb.storer(chunks, EMB, VECTOR_DB_PATH)

def main():
    rag = RAG(api_url=API_URL, model_name=MODEL_NAME)

    print("Loading knowledge base...")
    rag.load_knowledge_base(EMB, VECTOR_DB_PATH)

    ctx = Context(token_limit=4000, turns_to_leave=4)

    print("\n=== RAG TEST CLI ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User > ")
        if query.lower() in ["exit", "quit"]:
            break

        response = rag.run_turn(query, ctx)

        if response and "choices" in response:
            answer = response["choices"][0]["message"]["content"]
            
            print("\n--- Assistant ---")
            print(answer)
            print("-----------------")
            print(f"Token count: {ctx.token_count} / {ctx.token_limit}")
            print(f"History size: {len(ctx.history)} messages\n")
        else:
            print("\n[Error: No response received]\n")


if not __name__ == "__main__":
    main()