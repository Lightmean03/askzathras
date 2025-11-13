from typing import List, Union, Generator, Iterator
import os

from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("rag_pipeline module imported")
print("All imports successful")


class Pipeline:
    class Valves(BaseModel):
        """Pipeline configuration"""
        embedding_model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        faiss_index_path: str = "/app/faiss_index_"
        ollama_base_url: str = "http://ollama:11434"  # Use container name
        ollama_model: str = "llama3.2:3b"
        retrieval_k: int = 4
        score_threshold: float = 0.7
        fetch_k: int = 20
        device: str = "cpu"

    def __init__(self):
        """Initialize the pipeline"""
        # Required attributes for OpenWebUI
        self.type = "manifold"
        self.id = "rag_pipeline"
        self.name = "RAG Pipeline"
        self.valves = self.Valves()

        # Initialize components
        self.qa = None
        self.retriever = None
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.initialized = False

        print(f"RAG Pipeline initialized - ID: {self.id}, Name: {self.name}")

    @property
    def pipelines(self):
        """Return pipeline configuration for OpenWebUI"""
        return [
            {
                "id": self.id,
                "name": self.name,
            }
        ]

    def _load_rag_system(self):
        """Load the RAG system components"""
        if self.initialized:
            return True

        try:
            print("Loading RAG system...")

            # Check if FAISS index exists
            if not os.path.exists(self.valves.faiss_index_path):
                print(f"FAISS index not found at: {self.valves.faiss_index_path}")
                return False

            # Load embedding model
            model_kwargs = {"device": self.valves.device}
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.valves.embedding_model_name,
                model_kwargs=model_kwargs,
            )
            print("Embeddings loaded successfully")

            # Load vector store
            self.vectorstore = FAISS.load_local(
                self.valves.faiss_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("Vector store loaded successfully")

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.valves.retrieval_k,
                    "score_threshold": self.valves.score_threshold,
                    "fetch_k": self.valves.fetch_k,
                },
            )

            # Load LLM
            self.llm = OllamaLLM(
                model=self.valves.ollama_model,
                base_url=self.valves.ollama_base_url,
            )
            print("LLM loaded successfully")

            # Create QA chain
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
            )

            self.initialized = True
            print("RAG system ready!")
            return True

        except Exception as e:
            print(f"Error loading RAG system: {e}")
            self.initialized = False
            return False

    async def on_startup(self):
        """Called when the pipeline starts"""
        print(f"on_startup: {self.name}")
        # Lazy load in pipe, so nothing here
        pass

    async def on_shutdown(self):
        """Called when the pipeline shuts down"""
        print(f"on_shutdown: {self.name}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline function"""
        try:
            # Lazy load the RAG system if not initialized
            if not self.initialized:
                if not self._load_rag_system():
                    return (
                        "Error: RAG system could not be initialized. "
                        "Please check your FAISS index path and Ollama model availability."
                    )

            if not self.qa:
                return "Error: RAG system not properly initialized"

            print(f"Processing query: {user_message}")

            # Query the RAG system
            result = self.qa.invoke({"query": user_message})
            answer = result.get("result", "No answer generated")

            return answer

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return error_msg


# Test the pipeline when run directly (not used by pipelines server, but handy)
if __name__ == "__main__":
    print("Testing pipeline...")
    p = Pipeline()
    print(f"Pipeline created successfully: {p.id}")
    print(f"Pipelines property: {p.pipelines}")

