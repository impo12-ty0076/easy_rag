"""
Embedding model utilities for Easy RAG System.
This module provides functionality for embedding models used in vector databases.
"""

import os
import importlib
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple

class EmbeddingModel:
    """Base class for embedding models"""
    
    name = "Base Embedding Model"
    description = "Base embedding model class"
    dimension = 0
    required_packages = []
    api_key_env = None
    
    @classmethod
    def get_name(cls) -> str:
        """Get the name of the embedding model"""
        return cls.name
    
    @classmethod
    def get_description(cls) -> str:
        """Get the description of the embedding model"""
        return cls.description
    
    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimension of the embedding vectors"""
        return cls.dimension
    
    @classmethod
    def get_required_packages(cls) -> List[str]:
        """Get the required packages for this embedding model"""
        return cls.required_packages
    
    @classmethod
    def get_api_key_env(cls) -> Optional[str]:
        """Get the environment variable name for the API key, if needed"""
        return cls.api_key_env
    
    @classmethod
    def check_dependencies(cls) -> Tuple[bool, List[str]]:
        """
        Check if all required dependencies are installed
        Returns: (all_installed, missing_packages)
        """
        missing = []
        for package in cls.required_packages:
            try:
                importlib.import_module(package.split('==')[0])
            except ImportError:
                missing.append(package)
        
        return len(missing) == 0, missing
    
    @classmethod
    def check_api_key(cls) -> Tuple[bool, str]:
        """
        Check if the required API key is available
        Returns: (key_available, error_message)
        """
        if cls.api_key_env is None:
            return True, ""
        
        if cls.api_key_env in os.environ and os.environ[cls.api_key_env]:
            return True, ""
        
        return False, f"API key not found. Please set the {cls.api_key_env} environment variable."
    
    @classmethod
    def install_dependencies(cls) -> Tuple[bool, str]:
        """
        Install required dependencies
        Returns: (success, message)
        """
        _, missing = cls.check_dependencies()
        if not missing:
            return True, "All dependencies already installed"
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            return True, f"Successfully installed: {', '.join(missing)}"
        except subprocess.CalledProcessError as e:
            return False, f"Failed to install dependencies: {str(e)}"
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        Returns: List of embedding vectors
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string
        Returns: Embedding vector
        """
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI's text-embedding-ada-002 embedding model"""
    
    name = "OpenAI Ada 002"
    description = "OpenAI's text-embedding-ada-002 model (1536 dimensions)"
    dimension = 1536
    required_packages = ['openai']
    api_key_env = "OPENAI_API_KEY"
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI's API"""
        try:
            import openai
            
            # Set API key from environment
            openai.api_key = os.environ.get(self.api_key_env)
            
            # Create embeddings in batches (OpenAI has rate limits)
            batch_size = 20
            embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                batch_embeddings = [item["embedding"] for item in response["data"]]
                embeddings.extend(batch_embeddings)
            
            return embeddings
        except ImportError:
            raise ImportError("openai package is required. Please install it first.")
        except Exception as e:
            raise Exception(f"Error creating embeddings with OpenAI: {str(e)}")
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query using OpenAI's API"""
        try:
            import openai
            
            # Set API key from environment
            openai.api_key = os.environ.get(self.api_key_env)
            
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=query
            )
            
            return response["data"][0]["embedding"]
        except ImportError:
            raise ImportError("openai package is required. Please install it first.")
        except Exception as e:
            raise Exception(f"Error creating embedding with OpenAI: {str(e)}")


class HuggingFaceEmbedding(EmbeddingModel):
    """Sentence Transformers embedding model from Hugging Face"""
    
    name = "Sentence Transformers"
    description = "Sentence Transformers all-MiniLM-L6-v2 model (384 dimensions)"
    dimension = 384
    required_packages = ['sentence-transformers']
    model_name = "all-MiniLM-L6-v2"
    
    def __init__(self):
        self.model = None
    
    def _load_model(self):
        """Load the model if not already loaded"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers package is required. Please install it first.")
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed documents using Sentence Transformers"""
        self._load_model()
        
        # Create embeddings
        embeddings = self.model.encode(documents)
        
        # Convert numpy arrays to lists
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query using Sentence Transformers"""
        self._load_model()
        
        # Create embedding
        embedding = self.model.encode(query)
        
        # Convert numpy array to list
        return embedding.tolist()


class HuggingFaceMPNetEmbedding(EmbeddingModel):
    """MPNet embedding model from Hugging Face"""
    
    name = "MPNet"
    description = "Sentence Transformers all-mpnet-base-v2 model (768 dimensions)"
    dimension = 768
    required_packages = ['sentence-transformers']
    model_name = "all-mpnet-base-v2"
    
    def __init__(self):
        self.model = None
    
    def _load_model(self):
        """Load the model if not already loaded"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers package is required. Please install it first.")
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed documents using MPNet"""
        self._load_model()
        
        # Create embeddings
        embeddings = self.model.encode(documents)
        
        # Convert numpy arrays to lists
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query using MPNet"""
        self._load_model()
        
        # Create embedding
        embedding = self.model.encode(query)
        
        # Convert numpy array to list
        return embedding.tolist()


class CohereEmbedding(EmbeddingModel):
    """Cohere embedding model"""
    
    name = "Cohere Embed"
    description = "Cohere's embedding model (4096 dimensions)"
    dimension = 4096
    required_packages = ['cohere']
    api_key_env = "COHERE_API_KEY"
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed documents using Cohere's API"""
        try:
            import cohere
            
            # Set API key from environment
            co = cohere.Client(os.environ.get(self.api_key_env))
            
            # Create embeddings in batches (Cohere has rate limits)
            batch_size = 20
            embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                response = co.embed(
                    texts=batch,
                    model="embed-english-v3.0"
                )
                batch_embeddings = response.embeddings
                embeddings.extend(batch_embeddings)
            
            return embeddings
        except ImportError:
            raise ImportError("cohere package is required. Please install it first.")
        except Exception as e:
            raise Exception(f"Error creating embeddings with Cohere: {str(e)}")
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query using Cohere's API"""
        try:
            import cohere
            
            # Set API key from environment
            co = cohere.Client(os.environ.get(self.api_key_env))
            
            response = co.embed(
                texts=[query],
                model="embed-english-v3.0"
            )
            
            return response.embeddings[0]
        except ImportError:
            raise ImportError("cohere package is required. Please install it first.")
        except Exception as e:
            raise Exception(f"Error creating embedding with Cohere: {str(e)}")


# Dictionary of available embedding models
AVAILABLE_EMBEDDING_MODELS = {
    'openai': OpenAIEmbedding,
    'sentence-transformers': HuggingFaceEmbedding,
    'mpnet': HuggingFaceMPNetEmbedding,
    'cohere': CohereEmbedding
}


def get_available_embedding_models() -> List[Dict[str, Any]]:
    """Get a list of all available embedding models with their metadata"""
    models = []
    
    for model_id, model_class in AVAILABLE_EMBEDDING_MODELS.items():
        is_available, missing_packages = model_class.check_dependencies()
        api_key_available = True
        api_key_error = ""
        
        if model_class.get_api_key_env():
            api_key_available, api_key_error = model_class.check_api_key()
        
        models.append({
            'id': model_id,
            'name': model_class.get_name(),
            'description': model_class.get_description(),
            'dimension': model_class.get_dimension(),
            'required_packages': model_class.get_required_packages(),
            'api_key_env': model_class.get_api_key_env(),
            'is_available': is_available and api_key_available,
            'missing_packages': missing_packages,
            'api_key_error': api_key_error
        })
    
    return models


def get_embedding_model(model_id: str) -> EmbeddingModel:
    """Get an embedding model instance by ID"""
    if model_id not in AVAILABLE_EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding model: {model_id}")
    
    return AVAILABLE_EMBEDDING_MODELS[model_id]()


# Dictionary of available vector stores
AVAILABLE_VECTOR_STORES = {
    'chroma': {
        'name': 'Chroma',
        'description': 'Open-source embedding database with support for metadata filtering',
        'required_packages': ['chromadb'],
        'supports_metadata': True
    },
    'faiss': {
        'name': 'FAISS',
        'description': 'Facebook AI Similarity Search - efficient similarity search and clustering',
        'required_packages': ['faiss-cpu'],
        'supports_metadata': False
    }
}


def get_available_vector_stores() -> List[Dict[str, Any]]:
    """Get a list of all available vector stores with their metadata"""
    stores = []
    
    for store_id, store_info in AVAILABLE_VECTOR_STORES.items():
        # Check if required packages are installed
        missing_packages = []
        for package in store_info['required_packages']:
            try:
                importlib.import_module(package.split('==')[0])
            except ImportError:
                missing_packages.append(package)
        
        is_available = len(missing_packages) == 0
        
        stores.append({
            'id': store_id,
            'name': store_info['name'],
            'description': store_info['description'],
            'required_packages': store_info['required_packages'],
            'supports_metadata': store_info['supports_metadata'],
            'is_available': is_available,
            'missing_packages': missing_packages
        })
    
    return stores


def install_vector_store_dependencies(store_id: str) -> Tuple[bool, str]:
    """
    Install dependencies for a vector store
    Returns: (success, message)
    """
    if store_id not in AVAILABLE_VECTOR_STORES:
        return False, f"Unknown vector store: {store_id}"
    
    required_packages = AVAILABLE_VECTOR_STORES[store_id]['required_packages']
    
    # Check if packages are already installed
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package.split('==')[0])
        except ImportError:
            missing.append(package)
    
    if not missing:
        return True, "All dependencies already installed"
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        return True, f"Successfully installed: {', '.join(missing)}"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to install dependencies: {str(e)}"