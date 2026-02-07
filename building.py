import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from Knowledge_base import Knowledge_base

# --- Configuration ---
API_URL = "http://localhost:8000/v1/chat/completions"
VECTOR_DB_PATH = "./faiss_index-v5"
MARKDOWN_FOLDER = "./markdown_docs/"

# --- Initialize Embeddings ---
# Using standard HuggingFace embeddings as configured
emb2 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
EMB = emb2

# --- Main Execution Loop ---
if __name__ == "__main__":
    # 1. Get list of all markdown files
    if not os.path.exists(MARKDOWN_FOLDER):
        print(f"Error: Folder {MARKDOWN_FOLDER} does not exist.")
        sys.exit(1)

    all_files = [f for f in os.listdir(MARKDOWN_FOLDER) if f.endswith(".md")]
    total_files = len(all_files)
    
    print(f"Found {total_files} markdown files to process.")
    print("-" * 40)

    # 2. Iterate through files one by one
    for index, file_name in enumerate(all_files, 1):
        file_path = os.path.join(MARKDOWN_FOLDER, file_name)
        
        print(f"\n[{index}/{total_files}] Processing file: {file_name}...")

        try:
            # Initialize Knowledge_base with ONLY the current file
            # This ensures we process and store just this one file
            kb = Knowledge_base(list_file_paths=[file_path], api_url=API_URL)
            
            # Load, Split, and Store just this file
            documents = kb.loader()
            if not documents:
                print(f"Skipping {file_name} (empty or failed to load).")
                continue
                
            chunks = kb.splitter(documents)
            
            # The storer method (from your previous update) will:
            # - Load the existing DB at VECTOR_DB_PATH
            # - Add these new chunks
            # - Save it back to disk
            kb.storer(chunks, EMB, VECTOR_DB_PATH)
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            # Optional: ask to continue even if error occurred
        
        # 3. Interactive check after finishing the file
        # If it's the last file, don't ask, just finish.
        if index < total_files:
            user_input = input(f"\n> '{file_name}' stored. Continue to next file? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Stopping process.")
                break
        else:
            print("\nAll files processed successfully.")