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
        history = RAG.HISTORY
        if not history:
            return query
        
        if len(history) <= max_turns:
            hist_str = ""
            for e in history:
                hist_str += (
                    f"Utilisateur : {e['user']}\n"
                    f"Assistant : {e['ai']}\n"
                )

            augmented_prompt = prompts.HISTORY_PROMPT.format(
                history=hist_str,
                summary="pas de sommaire pour le momment.",
                question=query
            )
            return self.response_generator(augmented_prompt)["message"]["content"]

        recent = history[-max_turns:]
        hist_str = ""

        for e in recent:
            hist_str += (
                f"Utilisateur : {e['user']}\n"
                f"Assistant : {e['ai']}\n"
            )

        old_history = history[:-max_turns]

        summary_prompt = (
            "Résume de manière concise et factuelle la conversation suivante "
            "(max 200 tokens) :\n\n"
        )
        for e in old_history:
            summary_prompt += (
                f"Utilisateur : {e['user']}\n"
                f"Assistant : {e['ai']}\n"
            )

        summary = self.response_generator(summary_prompt)["message"]["content"]

        augmented_prompt = prompts.HISTORY_PROMPT.format(
            history=hist_str,
            summary=summary,
            question=query
        )

        return self.response_generator(augmented_prompt)["message"]["content"]


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


