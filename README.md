## Frontend

### Exécution du Frontend avec Docker

Pour exécuter l’application frontend à l’aide de Docker, exécutez la commande suivante depuis le répertoire `frontend` :

```bash
BACKEND_PORT=8072
cd frontend
docker build -t chat-interface . 
docker run -p 3000:3000 -e BACKEND_URL=http://127.0.0.1:${BACKEND_PORT} --network host chat-interface
```

## Backend
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
Pour éxecuter l'application:
```bash
python3 api.py
```
### docker avec docker
