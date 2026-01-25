import requests, json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import prompts



class RAG:

    
    HISTORY = []
    SUMMARY = ""

    def __init__(self, api_url: str, model_name: str):
        self.vectorstore = None
        self.Model_API_URL = api_url
        self.Model_NAME = model_name
        self.history = []

    def load_knowledge_base(self, path: str):
        self.vectorstore = FAISS.load_local(
            path,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True,
        )

    def retriever(self, query):
        docs = self.vectorstore.similarity_search(query, k=5)
        return docs

    def rewrite_query(self, query, max_turns=3):
        if RAG.SUMMARY == "":
            RAG.SUMMARY = "pas de sommaire pour le momment."

        history = RAG.HISTORY
        if not history:
            return query
            
        if len(history) > max_turns:
            e = history[-(max_turns + 1)]
            
            summary_prompt = (
                "Résume de manière concise et factuelle la conversation suivante "
                "(max 200 tokens) :\n\n"
                f"Ancien sommaire : {RAG.SUMMARY}\n"
                f"Nouvel échange à intégrer :\n"
                f"Utilisateur : {e['user']}\n"
                f"Assistant : {e['ai']}\n"
            )
            RAG.SUMMARY = self.response_generator(summary_prompt)["message"]["content"]

        recent = history[-max_turns:]
        hist_str = ""
        for e in recent:
            hist_str += (
                f"Utilisateur : {e['user']}\n"
                f"Assistant : {e['ai']}\n"
            )

        augmented_prompt = prompts.HISTORY_PROMPT.format(
            history=hist_str,
            summary=RAG.SUMMARY,
            question=query
        )

        return self.response_generator(augmented_prompt)["message"]["content"]

    def _update_incremental_summary(self, old_summary, new_turn):
        """Met à jour le sommaire existant avec un nouveau tour de parole (Appel LLM)."""
        update_prompt = (
            "Mets à jour le sommaire suivant en y intégrant les informations cruciales du nouvel échange. "
            "Le sommaire doit rester concis (max 150 mots).\n\n"
            f"Sommaire actuel : {old_summary}\n"
            f"Nouvel échange à intégrer :\nU: {new_turn['user']}\nA: {new_turn['ai']}\n\n"
            "Nouveau sommaire mis à jour :"
        )
        return self.response_generator(update_prompt)["message"]["content"]
        
    def prompt_augmentation(self, docs, query):
        rewritten_query = self.rewrite_query(query)
        print(rewritten_query)
        augmented_prompt = prompts.ZERO_SHOT_PROMPT_FR.format(
            context_str=docs,
            strict_instructions=prompts.INSTRUCTIONS_FR,
            query_str=rewritten_query,
        )
        #augmented_prompt = f"en se basant sur les documents suivants : {docs}, reponds a la question suivante : {rewritten_query}"
        return augmented_prompt

    def response_generator(self, augmented_prompt):
        print(augmented_prompt)
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.Model_NAME,
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": augmented_prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "stream": False,
        }
        response = requests.post(
            self.Model_API_URL, headers=headers, data=json.dumps(payload)
        )
        return response.json()


