# ChatEILCO

ChatEILCO est une application web intelligente basée sur une architecture RAG (Retrieval-Augmented Generation) conçue pour fournir des réponses précises et contextualisées concernant les informations académiques et administratives de l'école EILCO.

L'application combine un frontend interactif avec un backend puissant utilisant des modèles de langage (LLM) pour offrir une expérience conversationnelle fluide et informée.

## Frontend

### Exécution du Frontend avec Docker

Pour exécuter l’application frontend à l’aide de Docker, exécutez la commande suivante depuis le répertoire `frontend` :

```bash
BACKEND_PORT=8000
cd frontend
docker build -t chat-interface . 
docker run -p 3000:3000 -e BACKEND_URL=http://127.0.0.1:${BACKEND_PORT} --network host chat-interface
```

## Backend
```bash
cd backend
```
créer une envirenment virtual avec python3>10

```bash
python3.x -m venv venv

```
```bash
source venv/bin/activate
```
Installer les requirements:
```bash
pip intall -r requirements.txt
```

Pour convertir les docx qui sont en data/docx_files en pdf utiliser:
```bash
python3 src/
Pour éxecuter l'application:
```bash
python3 api.py
```
### Backend avec docker
```bash
docker build -t chat-eilco-backend .
```
```bash
docker run -p 8000:8000   -e LLM_API_URL=http://127.0.0.1:8080/v1/chat/completions   -v $(pwd)/data:/app/data --network host  chat-eilco-backend
```
