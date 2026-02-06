## Frontend

### Exécution du Frontend avec Docker

Pour exécuter l’application frontend à l’aide de Docker, exécutez la commande suivante depuis le répertoire `frontend` :

```bash
cd frontend
docker build -t chat-interface . && docker run -p 3000:3000 --name chat-interface -e BACKEND_URL=http://127.0.0.1:8072 --network host chat-interface
```

## Backend