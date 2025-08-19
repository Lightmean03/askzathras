# RAG System with OpenWebUI and Ollama

A complete Retrieval-Augmented Generation (RAG) system using Docker Compose with OpenWebUI, Ollama, and a custom RAG pipeline for custom question answering.

## Quick Start

```bash
# Clone or create project directory
cd pipeline_container
# remove exsisting volume 
docker volume rm pipeline_container_data
# Start the system
docker-compose up -d

# Wait for Ollama to start (about 30 seconds)
# Then pull the required model
docker exec ollama ollama pull llama3.2:3b

# Access the web interface
# Open http://localhost:3000 in your browser
# username: zathras@askzathras.com
# password Password1!
```

## Project Structure

```
rag-system/
├── pipelines/
│   └── rag_pipeline.py          # Custom RAG pipeline
├── data/                        # OpenWebUI data (auto-created)
├── faiss_index_/               # Your FAISS vector index
│   ├── index.faiss
│   ├── index.pkl
│   └── (other index files)
├── requirements.txt             # Python dependencies
├── docker-compose.yml          # Docker services configuration
├── setup-rag.ps1              # Setup script (optional)
└── README.md                   # This file
```

## Setup Instructions

### Prerequisites

- Docker Desktop installed and running
- Your FAISS index files ready


### Step 1: Start the System

```bash
# Start all containers
docker-compose up -d

# Check that all containers are running
docker-compose ps
```


### Step 2: Setup Ollama Model

```bash
# Wait for Ollama to be ready (about 30 seconds)
# Then pull the language model
docker exec ollama ollama pull llama3.2:3b

# Verify the model is installed
docker exec ollama ollama list
```

### Step 4: Access the Interface

1. Open your browser and go to `http://localhost:3000`
2. Sign in with zathras account in OpenWebUI
		- `zathras@askzathras.com`
		- `Password1!`
3. Select "RAG Pipeline" from the model dropdown
4. Start asking questions about your documents!

![[Pasted image 20250816151204.png]]
![[Pasted image 20250816151710.png]]
![[Pasted image 20250816151910.png]]- should look like that already

## Configuration

### Pipeline Settings

You can modify the RAG pipeline settings in `rag_pipeline.py` or in the pipeline section of the admin panel:

- **`embedding_model_name`**: The sentence transformer model for embeddings
- **`faiss_index_path`**: Path to your FAISS index (default: `/app/faiss_index_`)
- **`ollama_model`**: The Ollama model to use (default: `llama3.2:3b`)
- **`retrieval_k`**: Number of documents to retrieve (default: 4)
- **`score_threshold`**: Minimum similarity score (default: 0.7)

### Docker Services

- **OpenWebUI**: `http://localhost:3000` - Web interface
- **Pipelines**: `http://localhost:9099` - RAG pipeline API
- **Ollama**: `http://localhost:11434` - Language model API

## Daily Usage

### Starting the System

```bash
docker-compose up -d
```

### Stopping the System

```bash
docker-compose down
```

### Restarting (keeps all data)

```bash
docker-compose restart
```

### Checking Status

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs pipelines
docker-compose logs ollama
```

## Advanced Configuration

### Using Different Models

To use a different Ollama model:

1. Pull the model: `docker exec ollama ollama pull <model-name>`
2. Update `ollama_model` in `rag_pipeline.py` or admin panel 
3. Restart: `docker-compose restart pipelines`

### GPU Support

To enable GPU support for Ollama, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

###

### Clean Reset

```bash
# Stop and remove everything (keeps volumes)
docker-compose down

# Remove volumes too (complete reset)
docker-compose down -v

# Rebuild and start fresh
docker-compose up -d
```

