import requests, json
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import prompts
import tiktoken


class Context:
    def __init__(self, token_limit, turns_to_leave):
        self.summary = ""
        self.history = []
        self.token_limit = token_limit
        self.token_count = 0
        self.turns_to_leave = turns_to_leave


class RAG:

    enc = tiktoken.get_encoding("o200k_base")

    @staticmethod
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

    def load_knowledge_base(self, embeddings: Embeddings, path: str):
        self.vectorstore = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def retriever(self, query):
        docs = self.vectorstore.similarity_search(query, k=5)
        return docs

    def get_system_prompt(self, docs: list[str]):
        docs_text = [d.page_content for d in docs]
        return prompts.ZERO_SHOT_PROMPT_FR.format(
            context_str="\n\n".join(docs_text),
            strict_instructions=prompts.INSTRUCTIONS_FR,
        )

    def update_ctx(self, ctx: Context):
        if ctx.token_count < ctx.token_limit or len(ctx.history) <= ctx.turns_to_leave * 2:
            return

        messages_to_summarize = ctx.history[: -(ctx.turns_to_leave * 2)]
        if not messages_to_summarize:
            return

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages_to_summarize
        )

        system_prompt = (
            "Tu es un système de mise à jour de résumé de conversation.\n"
            "Tu dois maintenir un résumé concis, factuel et neutre.\n"
            "Tu n'ajoutes aucune interprétation ni information absente.\n"
            "Tu intègres uniquement les faits nouveaux."
        )

        user_prompt = (
            "Résumé existant:\n"
            f"{ctx.summary or 'Aucun.'}\n\n"
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

        response = self.chat_completion(messages, {"temperature": 0.2, "max_tokens": 512})

        if response and "choices" in response:
            ctx.summary = response["choices"][0]["message"]["content"]
            ctx.history = ctx.history[-(ctx.turns_to_leave * 2) :]
            ctx.token_count = RAG.count_chat_tokens(ctx.history)

    def rewrite_query(self, query: str, ctx: Context):

        if ctx.history:
            conversation_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in ctx.history
            )
        else:
            conversation_text= "pas de conversation"
        
        if ctx.summary == "":
            summary = "pas de sommaire"
        else:
            summary = ctx.summary

        system_prompt = prompts.SYSTEM_REWRITE_PROMPT
        #  (
        #     "Tu es un système de reformulation de requêtes pour un moteur de recherche documentaire (RAG) pour une ecole\n"
        #     "Les documents en générale concernent le reglement intérieur et les syllabus."
        #     "Ta tâche est de reformuler une question utilisateur pour qu'elle soit autonome, précise et adaptée à la recherche. Essayez d'augmenter la question plus detaillée pour maximiser la recherche\n"
        #     "Tu ne dois PAS répondre à la question.\n"
        #     "Tu ne dois PAS ajouter d'informations nouvelles.\n"
        #     "Tu produis UNIQUEMENT la requête reformulée."
        # )

        user_prompt = prompts.USER_REWRITE_PROMPT.format(
            summary=summary, history=conversation_text, question=query
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.chat_completion(
            messages, {"temperature": 0.1,"stream":False}
        )

        if response and "choices" in response:
            return response["choices"][0]["message"]["content"]
        
        return query

    def run_turn(self, query, ctx: Context):
        self.update_ctx(ctx)
        
        rewritten_query = self.rewrite_query(query, ctx)
        print(rewritten_query)
        
        docs = self.retriever(rewritten_query)
        sys_prompt = self.get_system_prompt(docs)
        user_entry = {"role": "user", "content": query}
        messages = [{"role": "system", "content": sys_prompt}] + ctx.history + [user_entry]
        
        response = self.chat_completion(messages, {"temperature": 0.1, "stream": False,"model":self.Model_NAME})
        print(sys_prompt)
        
        if response and "choices" in response:
            answer = response["choices"][0]["message"]["content"]
            ctx.history.extend([user_entry, {"role": "assistant", "content": answer}])
            ctx.token_count = RAG.count_chat_tokens(ctx.history)
        
        return response

    def chat_completion(self, messages, options):
        headers = {"Content-Type": "application/json"}
        payload = {"messages": messages, **options}
        try:
            response = requests.post(
                self.Model_API_URL, headers=headers, data=json.dumps(payload), timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API Error: {e}")
            return None
