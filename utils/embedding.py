"""
Unified Embedding Client for HuggingFace models
Supports Snowflake Arctic Embed and other sentence transformers
"""

from typing import List, Optional
import os


class EmbeddingClient:
    """
    Universal embedding client supporting multiple providers
    """
    
    def __init__(
        self,
        provider: str = "huggingface",
        model: str = "Snowflake/snowflake-arctic-embed-m",
        api_key: Optional[str] = None
    ):
        """
        Initialize embedding client
        
        Args:
            provider: 'huggingface' or 'openai'
            model: Model identifier
            api_key: API key/token
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate embedding backend"""
        if self.provider == "huggingface":
            self._initialize_huggingface()
        elif self.provider == "openai":
            self._initialize_openai()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _initialize_huggingface(self):
        """Initialize HuggingFace Sentence Transformer"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading embedding model: {self.model}...")
            
            # Load model (will download if not cached)
            self._client = SentenceTransformer(
                self.model,
                device='cpu'  # Use 'cuda' if GPU available
            )
            
            print(f"✓ Model loaded successfully (dim: {self._client.get_sentence_embedding_dimension()})")
            
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            raise
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(api_key=self.api_key)
            print(f"✓ OpenAI client initialized with model: {self.model}")
            
        except Exception as e:
            print(f"Error initializing OpenAI: {e}")
            raise
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []
        
        if self.provider == "huggingface":
            return self._embed_huggingface(texts)
        elif self.provider == "openai":
            return self._embed_openai(texts)
    
    def _embed_huggingface(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace"""
        try:
            # Batch encoding
            embeddings = self._client.encode(
                texts,
                batch_size=32,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )
            
            # Convert to list format
            return embeddings.tolist()
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            # OpenAI has batch limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self._client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            
            return all_embeddings
            
        except Exception as e:
            print(f"Error with OpenAI embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self.provider == "huggingface":
            return self._client.get_sentence_embedding_dimension()
        elif self.provider == "openai":
            # OpenAI dimensions
            if "text-embedding-3-large" in self.model:
                return 3072
            elif "text-embedding-3-small" in self.model:
                return 1536
            else:  # ada-002
                return 1536
        return 0


# Presets for popular models
EMBEDDING_PRESETS = {
    "snowflake": {
        "provider": "huggingface",
        "model": "Snowflake/snowflake-arctic-embed-m",
        "dim": 768
    },
    "bge-m3": {
        "provider": "huggingface",
        "model": "BAAI/bge-m3",
        "dim": 1024
    },
    "minilm": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384
    },
    "mpnet": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768
    },
    "openai-small": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dim": 1536
    },
    "openai-large": {
        "provider": "openai",
        "model": "text-embedding-3-large",
        "dim": 3072
    }
}


def create_embedding_client(preset: str = "snowflake", api_key: Optional[str] = None) -> EmbeddingClient:
    """
    Create embedding client from preset
    
    Args:
        preset: One of the preset names (snowflake, bge-m3, minilm, etc.)
        api_key: Optional API key
        
    Returns:
        Configured EmbeddingClient
    """
    if preset not in EMBEDDING_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(EMBEDDING_PRESETS.keys())}")
    
    config = EMBEDDING_PRESETS[preset]
    
    return EmbeddingClient(
        provider=config["provider"],
        model=config["model"],
        api_key=api_key
    )


if __name__ == "__main__":
    # Test embedding
    client = EmbeddingClient(
        provider="huggingface",
        model="Snowflake/snowflake-arctic-embed-m"
    )
    
    texts = [
        "I have 5 years of Python experience",
        "Strong background in data engineering"
    ]
    
    embeddings = client.embed(texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Dimension: {len(embeddings[0])}")