# ChatEILCO

ChatEILCO est une application web intelligente basée sur une architecture RAG (Retrieval-Augmented Generation) conçue pour fournir des réponses précises et contextualisées concernant les informations académiques et administratives de l'école EILCO.

L'application combine un frontend interactif avec un backend puissant utilisant des modèles de langage (LLM) pour offrir une expérience conversationnelle fluide et informée.

## Frontend

### Exécution du Frontend avec Docker

Pour exécuter l’application frontend à l’aide de Docker, exécutez la commande suivante depuis le répertoire `frontend` :

```bash
cd frontend
docker build -t chat-interface . 
docker run -p 3000:3000 -e BACKEND_URL=http://127.0.0.1:8000 --network host chat-interface
```

## Backend
Changer le repertoire:
```bash
cd backend
```
créer une envirenment virtuel avec python3>10: 

```bash
python3.x -m venv venv

```
Activer l'environement virtuel: 
```bash
source venv/bin/activate
```

Installer les requirements:
```bash
pip intall -r requirements.txt
```

Pour convertir les docx qui sont en data/docx_files en pdf utiliser:
```bash
python3 src/convert_word_to_pdf.py
```
Pour éxecuter l'applicationen mode développement:
```bash
python3 api.py
```
### Backend avec docker
```bash
docker build -t chat-eilco-backend .
```
```bash
docker run -p 8000:8000   -e LLM_API_URL=http://127.0.0.1:8080/v1/chat/completions   -v $(pwd)/data:/data --network host  chat-eilco-backend
```
