import requests, json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import prompts
import tiktoken



class RAG:

    enc = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def count_chat_tokens(messages):
        total = 0
        for m in messages:
            total += len(RAG.enc.encode(m["content"]))
        return total

    def __init__(self, api_url: str, model_name: str, rewrite_query: bool = False):
        self.rewrite = rewrite_query
        self.vectorstore = None
        self.Model_API_URL = api_url
        self.Model_NAME = model_name

    def load_knowledge_base(self, path: str):
        self.vectorstore = FAISS.load_local(
            path,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True,
        )

    def retriever(self, query):
        docs = self.vectorstore.similarity_search(query, k=5)
        return docs
    
    def get_system_prompt(self, docs:list[str]):
        docs_text = [d.page_content for d in docs]
        
        return prompts.ZERO_SHOT_PROMPT_FR.format(
            context_str="\n\n".join(docs_text), # Now joining strings, not objects
            strict_instructions=prompts.STRICT_INSTRUCTIONS_FR
        )
    
    def update_summary(self, history: list[dict], current_summary: str, max_turns: int = 3):

        if len(history) <= max_turns:
            return None

        messages_to_summarize = history[:-max_turns]

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages_to_summarize
        )

        system_prompt = (
            "Tu es un système de mise à jour de résumé de conversation.\n"
            "Tu dois maintenir un résumé concis, factuel et neutre.\n"
            "Tu n'ajoutes aucune interprétation ni information absente.\n"
            "Tu intègres uniquement les faits nouveaux."
        )

        user_prompt = (
            "Résumé existant:\n"
            f"{current_summary or 'Aucun.'}\n\n"
            "Nouveaux échanges à intégrer:\n"
            f"{conversation_text}\n\n"
            "INSTRUCTION:\n"
            "Mets à jour le résumé en tenant compte uniquement des nouveaux échanges.\n\n"
            "Résumé mis à jour:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.chat_completion(
            messages,
            {
                "temperature": 0.2,
                "max_tokens": 512,
            }
        )
    
    def rewrite_query(self, query: str, summary: str, history: list[dict], last_turns: int):
        if not history:
            return None
        
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history[-last_turns:]
        )

        system_prompt = (
            "Tu es un système de reformulation de requêtes pour un moteur de recherche documentaire (RAG).\n"
            "Ta tâche est de reformuler une question utilisateur pour qu’elle soit autonome, précise et adaptée à la recherche.\n"
            "Tu ne dois PAS répondre à la question.\n"
            "Tu ne dois PAS ajouter d’informations nouvelles.\n"
            "Tu produis UNIQUEMENT la requête reformulée."
        )

        user_prompt = (
            prompts.REWRITE_PROMPT.format(summary=summary,history=conversation_text, question=query)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.chat_completion(
            messages,
            {
                "temperature": 0.1,
                "max_tokens": 1028,
                "stop": ["\n"]
            }
        )
    
    def run_turn(self, query, summary, history, max_turns):
        response = self.update_summary(history, summary, max_turns)
        if response is None:
            new_summary = "Pas de resumé précédent"
        else:
            new_summary = response["choices"][0]["message"]["content"]
        print(new_summary)
        response = self.rewrite_query(query, new_summary, history, max_turns)
        rewritten_query = response["choices"][0]["message"]["content"] if response is not None else query
        print(rewritten_query)
        docs = self.retriever(rewritten_query)
        sys_prompt = self.get_system_prompt(docs)
        messages = [{'role':"system","content":sys_prompt}] + history[-max_turns:] + [{"role":"user","content":query}]
        return self.chat_completion(messages,{"temperature": 0.1,"stream":False})


    def chat_completion(self, messages, options):
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": messages,
            **options
        }
        response = requests.post(
            self.Model_API_URL, headers=headers, data=json.dumps(payload)
        )
        return response.json()

    def response_generator(self, messages, options):
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": messages,
            **options
        }
        response = requests.post(
            self.Model_API_URL, headers=headers, data=json.dumps(payload)
        )
        return response.json()
