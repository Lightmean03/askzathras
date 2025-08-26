# RAG System with OpenWebUI and Ollama

A complete Retrieval-Augmented Generation (RAG) system using Docker Compose with OpenWebUI, Ollama, and a custom RAG pipeline for document question answering.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Configuration](#configuration)
5. [Daily Usage](#daily-usage)
6. [Generating FAISS Index from PDF](#generating-faiss-index-from-pdf)
7. [Advanced Configuration](#advanced-configuration)
8. [Support](#support)

## Quick Start

```bash
# Clone or create project directory
cd pipeline_container
# remove existing volume 
docker volume rm pipeline_container_data
# Start the system
docker-compose up -d

# Wait for Ollama to start (about 30 seconds)
# Then pull the required model
docker exec ollama ollama pull llama3.2:3b

# Access the web interface
# Open http://localhost:3000 in your browser
# username: zathras@askzathras.com
# password: Password1!
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
├── content/                     # Source documents for indexing
│   └── book.pdf                 # Your PDF document
├── requirements.txt             # Python dependencies
├── docker-compose.yml          # Docker services configuration
├── create_index.py             # Script to generate FAISS index
├── setup-rag.ps1              # Setup script (optional)
└── README.md                   # This file
```

## Setup Instructions

### Prerequisites

- Docker Desktop installed and running
- Your FAISS index files ready (or PDF to generate index from)

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

### Step 3: Access the Interface

1. Open your browser and go to `http://localhost:3000`
2. Sign in with zathras account in OpenWebUI
   - `zathras@askzathras.com`
   - `Password1!`
3. Select "RAG Pipeline" from the model dropdown
4. Start asking questions about your documents!



![(https://github.com/Lightmean03/askzathras/blob/main/screenshots/login.png)]()


![screenshots/pipelin.png]()


![(https://github.com/Lightmean03/askzathras/blob/main/screenshots/example.png)]()

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

## Generating FAISS Index from PDF

The container system includes a pre-generated FAISS index for demonstration purposes. However, you can create your own index from any PDF document.

### Prerequisites for Index Generation

- Python 3.8+ installed locally
- Required Python packages (see requirements below)

### Step 1: Install Dependencies

```bash
pip install langchain-ollama langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers pypdf
```

### Step 2: Prepare Your Document

1. Create a `content/` directory in your project folder
2. Place your PDF file in the directory (e.g., `content/your-document.pdf`)
3. Update the file path in the index generation script

### Step 3: Modify the Index Generation Script

Update the document path in `create_index.py`:

```python
# Change this line to point to your PDF file
loader = PyPDFLoader("./content/your-document.pdf")
```

### Step 4: Generate the Index

```bash
# Run the index generation script
python create_index.py
```

The script will:
- Load and parse your PDF document
- Split the text into chunks
- Generate embeddings for each chunk
- Create and save the FAISS index to `faiss_index_/`
- Run diagnostic tests to verify the index


### Changing Document Source

To use a different document:

1. **Replace the PDF**: Place your new PDF in the `content/` directory
2. **Update the path**: Modify the file path in `create_index.py`
3. **Regenerate index**: Run `python create_index.py`
4. **Restart containers**: Run `docker-compose restart pipelines`

**Important Note**: The current system includes a pre-generated index for demonstration. You must replace this with your own document and regenerate the index, as the original document is not publicly available.

### GPU Acceleration (Linux Only)

To use CUDA for faster embedding generation on Linux systems:

1. **Install CUDA-enabled packages**:
   ```bash
   pip uninstall faiss-cpu
   pip install faiss-gpu
   ```

2. **Update device setting in `create_index.py`**:
   ```python
   # Change this line
   model_kwargs = {"device": "cuda"}  # was "cpu"
   ```

3. **Ensure NVIDIA Docker support** is enabled in your Docker setup

### Customizing Index Generation

You can modify several parameters in `create_index.py`:

```python
# Text splitting parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,        # Adjust chunk size
    chunk_overlap=50,      # Adjust overlap between chunks
    separators=["\n\n", "\n", " ", ""]
)

# Embedding model selection
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Change model

# Retrieval parameters
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Number of chunks to retrieve
)
```

### Testing Your Generated Index

The index generation script includes comprehensive diagnostics:

- **Document parsing verification**: Checks if PDF loaded correctly
- **Text splitting analysis**: Verifies chunk sizes and content
- **Embedding creation**: Tests vector store functionality
- **Retrieval testing**: Runs sample queries against your content

### Common Issues

- **Empty pages detected**: Some PDFs have formatting that creates empty pages - this is usually normal
- **Short chunks**: Very short chunks might indicate formatting issues in the source PDF
- **CUDA errors**: Ensure you have compatible NVIDIA drivers and CUDA toolkit installed
- **Memory issues**: Large documents may require more RAM - consider reducing chunk_size

## Advanced Configuration

### Using Different Models

To use a different Ollama model:

1. Pull the model: `docker exec ollama ollama pull <model-name>`
2. Update `ollama_model` in `rag_pipeline.py` or admin panel 
3. Restart: `docker-compose restart pipelines`

### GPU Support for Ollama

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

## Support

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs pipelines
docker-compose logs ollama
docker-compose logs open-webui

# Follow logs in real-time
docker-compose logs -f pipelines
```

### Clean Reset

```bash
# Stop and remove everything (keeps volumes)
docker-compose down

# Remove volumes too (complete reset)
docker-compose down -v

# Rebuild and start fresh
docker-compose up -d
```
