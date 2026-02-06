
from typing import List
import requests, json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
class RAG:
    def __init__(self, model_api_url: str="http://localhost:8080/v1/chat/completions", model_name: str="mistral", retriever=None, prompt=None):
        self.vectorstore = None
        self.Model_API_URL = model_api_url
        self.Model_NAME = model_name
        self.retriever = retriever
        self.prompt = prompt

    def augment_prompt(self, query: str, relevant_docs: str=None):
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        augmented_prompt = self.prompt.format(
            context=context,
            question=query
        )
        return augmented_prompt
    
    def condense_query_for_retriever(self, user_query: str, chat_history: List[dict] = None) -> str:
        """
        Condense a user query using chat history context for better retriever performance.
        Uses LLM to rewrite query with context from conversation.
        
        Args:
            user_query (str): The current user question
            chat_history (List[dict], optional): Previous messages in conversation
            
        Returns:
            str: Condensed/rewritten query optimized for retrieval
        """
        # If no chat history, return original query
        if not chat_history or len(chat_history) == 0:
            return user_query
        
        # Build context from last 3 messages
        recent_context = chat_history[-3:] if len(chat_history) > 3 else chat_history
        context_str = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_context])
        
        condense_prompt = f"""Based on this conversation context, rewrite the latest user question as a standalone, clear question that incorporates relevant context from the conversation.

Context:
{context_str}

User's latest question: {user_query}

Rewritten standalone question:"""
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.Model_NAME,
            "messages": [
                {"role": "system", "content": "Tu es un assistant qui reformule les questions pour les clarifier et les rendre autonomes. Réponds uniquement avec la question reformulée, sans explications supplémentaires."},
                {"role": "user", "content": condense_prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False
        }
        
        try:
            response = requests.post(self.Model_API_URL, headers=headers, data=json.dumps(payload), timeout=10)
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                condensed = result['choices'][0]['message']['content'].strip()
                return condensed if condensed else user_query
            else:
                return user_query
        except Exception as e:
            print(f"Error condensing query: {str(e)}")
            return user_query
    

    def response_generator(self, prompt: str, chat_history: List[dict] = None):
        """Generate response using the LLM with the configured system prompt.
        
        Args:
            prompt (str): The augmented prompt with context and question
            chat_history (List[dict], optional): Last 6 messages from chat history to include context
        """
        headers = {"Content-Type": "application/json"}
        
        # Build messages list with chat history
        messages = []
        
        # Add last 6 messages from chat history if provided
        if chat_history:
            # Keep only the last 6 messages
            recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
            messages.extend(recent_messages)
        
        # Add system prompt and current prompt
        messages.append({"role": "system", "content": "Tu es ChatEILCO, l'assistant virtuel de l'EILCO. Réponds toujours en français."})
        messages.append({"role": "user", "content": str(prompt)})
        
        payload = {
            "model": self.Model_NAME,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
        response = requests.post(self.Model_API_URL, headers=headers, data=json.dumps(payload))
        return response.json()

    def clip_text(self, text: str, max_length: int=100):
        """Clip text to a maximum length, adding ellipsis if it exceeds the limit.
        Args:            text (str): The text to be clipped.
            max_length (int): The maximum allowed length of the text. Defaults to 100.
        Returns:            str: The clipped text, with ellipsis if it was truncated.
        """
        if len(text) <= max_length:
            return text
        else:
            return text[:max_length]+"..."
        
    
    def sources_as_list(self, docs) -> List[str]:
        sources_list = []
        for doc in docs:
            doc_content = self.clip_text(doc.page_content, max_length=500)
            doc_metadata = doc.metadata
            filename = "unknown"
            page_number = "unknown"
            
            try:
                if 'dl_meta' in doc_metadata:
                    dl_meta = doc_metadata['dl_meta']
                    
                    # Get filename from origin
                    if 'origin' in dl_meta and isinstance(dl_meta['origin'], dict):
                        filename = dl_meta['origin'].get('filename', 'unknown')
                    
                    # Get page number from first doc_item
                    if 'doc_items' in dl_meta and isinstance(dl_meta['doc_items'], list):
                        if len(dl_meta['doc_items']) > 0:
                            first_item = dl_meta['doc_items'][0]
                            if 'prov' in first_item and isinstance(first_item['prov'], list):
                                if len(first_item['prov']) > 0:
                                    page_number = first_item['prov'][0].get('page_no', 'unknown')
            except (KeyError, TypeError, IndexError):
                pass
            
            source_info = {"Filename": filename, "Page": page_number, "Content": doc_content}
            sources_list.append(source_info)
        
        return sources_list