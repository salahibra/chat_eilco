CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  role TEXT,           -- user | assistant | system
  content TEXT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
