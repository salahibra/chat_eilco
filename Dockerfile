FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (excluding torch and faiss-gpu for lighter image)
RUN pip install --no-cache-dir \
    docling \
    langchain \
    sentence-transformers \
    faiss-cpu \
    fastapi \
    uvicorn[standard] \
    pydantic \
    langchain_community \
    langchain_docling \
    langgraph

# Copy application files
COPY api.py .
COPY config.py .
COPY RAG.py .
COPY Knowledge_base.py .
COPY context_merger.py .
COPY query_router.py .
COPY eilco_prompts.py .
COPY file.sql .

# Copy data directory (vectorstore)
COPY data/ ./data/

# Copy test files for knowledge base
COPY test/ ./test/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LLM_API_URL=http://host.docker.internal:8080/v1/chat/completions

# Start FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
