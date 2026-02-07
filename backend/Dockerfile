FROM nvidia/cuda:12-runtime

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/api.py .
COPY src/config.py .
COPY src/RAG.py .
COPY src/Knowledge_base.py .
COPY src/context_merger.py .
COPY src/query_router.py .
COPY src/eilco_prompts.py .
COPY src/file.sql .



# Create required directories
RUN mkdir -p ./data/



# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LLM_API_URL=http://host.docker.internal:8080/v1/chat/completions

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
