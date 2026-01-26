import os
from RAG import RAG 
import tiktoken # Adjust import if your file name is different

API_URL = "http://localhost:8080/v1/chat/completions"  # change to your API
MODEL_NAME = "gemma"  # change if needed
VECTOR_DB_PATH = "./faiss_index-v2"  # path to your FAISS folder

def main():
    rag = RAG(api_url=API_URL, model_name=MODEL_NAME)

    print("Loading knowledge base...")
    rag.load_knowledge_base(VECTOR_DB_PATH)

    history = []
    summary = ""

    max_turns = 4

    print("\n=== RAG TEST CLI ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User > ")
        if query.lower() in ["exit", "quit"]:
            break

        # Run RAG turn
        response = rag.run_turn(query, summary, history, max_turns)

        # Extract answer (depends on your API)
        answer = response["choices"][0]["message"]["content"]

        print("\n--- Assistant ---")
        print(answer)
        print("-----------------\n")

        # Update history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        # Update summary (only if needed)
        if len(history) > max_turns:
            summary_resp = rag.update_summary(history, summary, max_turns)
            if summary_resp is not None:
                summary = summary_resp["choices"][0]["message"]["content"]

        # Keep history size bounded
        history = history[-(max_turns * 2):]


if __name__ == "__main__":
    main()
