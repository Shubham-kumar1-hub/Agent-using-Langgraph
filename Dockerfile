# ---------------------------------------------------------------
#  Dockerfile
#  Project : Multi-Tool AI Agent with Human-in-the-Loop
#  Python  : 3.12-slim (lightweight official image)
# ---------------------------------------------------------------

# Step 1
FROM python:3.12-slim

# Step 2: Setting the working directory inside the container
#         All my project files will live here inside Docker
WORKDIR /app

# Step 3: Installing system dependencies needed by some Python packages
#         - build-essential : needed to compile some pip packages
#         - curl            : useful for health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements.txt first 

COPY requirements.txt .

# Step 5: Installing all Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of my project files into the container
COPY . .

# Step 7: Creating the folders my app needs at runtime
#         faiss_indexes : where FAISS vector stores are saved
RUN mkdir -p faiss_indexes

# Step 8:
#         (This is Streamlit's default port)
EXPOSE 8501

# Step 9: Health check - Docker will ping this every 30s
#         to make sure the app is still running
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Step 10: The command that runs when the container starts
#          --server.address=0.0.0.0 makes it accessible outside the container
#          --server.port=8501 sets the port
#          --server.headless=true disables the browser auto-open inside Docker
CMD ["streamlit", "run", "Agent_frontend.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
