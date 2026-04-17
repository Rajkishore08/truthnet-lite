# ==========================================
# STAGE 1: VITE REACT UI BUILD
# ==========================================
FROM node:18 AS frontend-build
WORKDIR /app/frontend/truthnet-ui

# Pull only explicit JS dependencies ensuring caching
COPY frontend/truthnet-ui/package*.json ./
RUN npm install

# Build compiled optimized React static elements
COPY frontend/truthnet-ui/ ./
RUN npm run build 

# ==========================================
# STAGE 2: PRODUCTION MPI ENVIRONMENT
# ==========================================
FROM python:3.10-slim

# Root OS setup natively instantiating OpenMPI & C++ dependencies natively required by `mpi4py`
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pull Explicit PyPI Requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Manually cache deep AI parsing matrices globally ensuring containers boot instantaneously
RUN python -m nltk.downloader punkt punkt_tab stopwords
    
# Copy the pure backend logic entirely natively
COPY backend/ ./backend/

# Dynamically scrape the React compiled Static interface natively into UI mounting paths
COPY --from=frontend-build /app/frontend/truthnet-ui/dist /app/frontend/truthnet-ui/dist

# Expose Render standard Cloud Port Mapping
EXPOSE 10000

# Bind Uvicorn natively capturing external traffic natively using zero-config variables
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
