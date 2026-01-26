import os
from RAG import RAG, Context

API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "gemma"
VECTOR_DB_PATH = "./faiss_index-v2"

def main():
    rag = RAG(api_url=API_URL, model_name=MODEL_NAME)

    print("Loading knowledge base...")
    rag.load_knowledge_base(VECTOR_DB_PATH)

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


if __name__ == "__main__":
    main()