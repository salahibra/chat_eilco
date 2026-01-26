STRICT_INSTRUCTIONS_FR = """* **Pas de Connaissances Extérieures :** Répondez uniquement à partir du contexte fourni.
* **Pas d'Hallucinations :** Si la réponse n'est pas dans le contexte, répondez : "Je suis désolé, je ne peux pas répondre".
* **Ton :** Professionnel, clair et direct."""


INSTRUCTIONS_FR = """1. Lisez attentivement la question pour comprendre l'intention.
2. Cherchez la réponse uniquement dans le contexte fourni.
3. Répondez clairement et concisément, en utilisant uniquement le contexte.
4. Respectez la langue de la question utilisateur.
5. Pas de Connaissances Extérieures :** Répondez uniquement à partir du contexte fourni.
6. Si la réponse n'est pas dans le contexte, répondez : "Je suis désolé, je ne peux pas répondre".
7. Ton :** Professionnel, clair et direct."""

ZERO_SHOT_PROMPT_FR = """### RÔLE
Vous êtes un assistant expert, source unique de vérité, répondant uniquement à partir du contexte fourni.

### CONTEXTE
--- DÉBUT ---
{context_str}
--- FIN ---

### CONTRAINTES
{strict_instructions}
"""


REWRITE_PROMPT = """
### RÔLE
Tu es un expert en ingénierie de requêtes (Prompt Engineering). Ton rôle est de transformer une question utilisateur issue d'une conversation en une question autonome et précise.

### CONTEXTE
- Résumé : {summary}
- Historique : {history}

### MISSION
À partir de la "Question Originale", génère une question unique qui :
1. Est parfaitement compréhensible sans l'historique (résolution des pronoms : remplace "il", "ça", "ce projet" par les noms réels mentionnés plus haut).
2. Est concise et va droit au but.
3. Conserve l'intention initiale et la langue de l'utilisateur.

### CONTRAINTES STRICTES
- Ne réponds PAS à la question.
- Ne donne aucune explication ou introduction (ex: ne dis pas "Voici la question reformulée...").
- Renvoie uniquement le texte de la question finale.

### QUESTION ORIGINALE :
{question}

### QUESTION REFORMULÉE :
"""



# HISTORY_PROMPT = """
# À partir du résumé et de l’historique de la conversation ci-dessous,
# génère une nouvelle question autonome qui :
# - intègre le contexte pertinent,
# - clarifie l’intention de l’utilisateur,
# - peut être comprise sans l’historique.

# ### Résumé :
# {summary}

# ### Historique :
# {history}

# ### Question originale :
# {question}

# ### Question contextualisée :
# """
