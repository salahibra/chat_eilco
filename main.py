from RAG import RAG
from Knowledge_base import Knowledge_base
import os

def initialize_rag():
    """Initialize the RAG system with knowledge base."""
    vectordb_path = "faiss_index"    
    rag = RAG(
        "http://localhost:8080/v1/chat/completions",
        "gemma",
    )
    rag.load_knowledge_base(vectordb_path)
    
    return rag

def chat_loop(rag):
    """Main interactive chat loop."""
    print("=" * 60)
    print("Interactive RAG Chat")
    print("=" * 60)
    print("Type 'exit', 'quit', or 'q' to end the conversation")
    print("Type 'clear' to clear chat history")
    print("Type 'history' to view conversation history")
    print("=" * 60)
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'clear':
                RAG.HISTORY.clear()
                print("\n[History cleared]\n")
                continue
            
            if user_input.lower() == 'history':
                print("\n--- Conversation History ---")
                if not RAG.HISTORY:
                    print("(empty)")
                else:
                    for i, turn in enumerate(RAG.HISTORY, 1):
                        print(f"\n{i}. User: {turn['user']}")
                        print(f"   AI: {turn['ai'][:100]}..." if len(turn['ai']) > 100 else f"   AI: {turn['ai']}")
                print("--- End of History ---\n")
                continue
            
            if not user_input:
                continue
            
            # Process query
            print("\nAI: ", end="", flush=True)
            docs = ""#rag.retriever(user_input)
            #augmented_prompt = rag.prompt_augmentation(docs, user_input)
            response = rag.response_generator({"system":"","user":user_input})
            ai_content = response['choices'][0]['message']['content']
            
            # Update history
            RAG.HISTORY.append({"user": user_input, "ai": ai_content})
            
            # Print response
            print(f"{ai_content}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error: {e}]\n")

if __name__ == "__main__":
    try:
        rag = initialize_rag()
        chat_loop(rag)
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")