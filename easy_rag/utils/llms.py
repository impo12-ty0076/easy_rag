"""
Utility functions for LLM configuration and management.
"""
from typing import Dict, List, Any, Optional, Tuple
import os
from dotenv import load_dotenv

# Define available LLMs
AVAILABLE_LLMS = {
    "openai-gpt-3.5-turbo": {
        "name": "OpenAI GPT-3.5 Turbo",
        "description": "OpenAI's GPT-3.5 Turbo model - good balance of performance and cost",
        "requires_api_key": True,
        "api_key_name": "OPENAI_API_KEY",
        "icon": "cloud",
        "performance": "High",
        "latency": "Low",
        "category": "api",
        "max_tokens": 4096,
        "cost": "Low"
    },
    "openai-gpt-4": {
        "name": "OpenAI GPT-4",
        "description": "OpenAI's GPT-4 model - high performance but higher cost",
        "requires_api_key": True,
        "api_key_name": "OPENAI_API_KEY",
        "icon": "cloud",
        "performance": "Very High",
        "latency": "Medium",
        "category": "api",
        "max_tokens": 8192,
        "cost": "High"
    },
    "anthropic-claude-instant": {
        "name": "Anthropic Claude Instant",
        "description": "Anthropic's Claude Instant model - fast and efficient",
        "requires_api_key": True,
        "api_key_name": "ANTHROPIC_API_KEY",
        "icon": "cloud",
        "performance": "High",
        "latency": "Low",
        "category": "api",
        "max_tokens": 100000,
        "cost": "Low"
    },
    "anthropic-claude-3-opus": {
        "name": "Anthropic Claude 3 Opus",
        "description": "Anthropic's Claude 3 Opus model - highest performance",
        "requires_api_key": True,
        "api_key_name": "ANTHROPIC_API_KEY",
        "icon": "cloud",
        "performance": "Very High",
        "latency": "Medium",
        "category": "api",
        "max_tokens": 200000,
        "cost": "High"
    },
    "llama-2-7b": {
        "name": "Llama 2 (7B)",
        "description": "Meta's Llama 2 model (7B parameters) - good for local deployment",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "Medium",
        "latency": "Medium",
        "category": "local",
        "download_required": True,
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "quantization": "None",
        "ram_required": "16GB+"
    },
    "llama-2-7b-q4": {
        "name": "Llama 2 (7B) - 4-bit Quantized",
        "description": "Meta's Llama 2 model with 4-bit quantization - reduced memory requirements",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "Medium",
        "latency": "Medium",
        "category": "local",
        "download_required": True,
        "model_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "quantization": "Q4_K_M",
        "ram_required": "8GB+"
    },
    "mistral-7b": {
        "name": "Mistral (7B)",
        "description": "Mistral AI's 7B parameter model - excellent performance for its size",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "Medium-High",
        "latency": "Medium",
        "category": "local",
        "download_required": True,
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "quantization": "None",
        "ram_required": "16GB+"
    },
    "mistral-7b-q4": {
        "name": "Mistral (7B) - 4-bit Quantized",
        "description": "Mistral AI's 7B model with 4-bit quantization - reduced memory requirements",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "Medium-High",
        "latency": "Medium",
        "category": "local",
        "download_required": True,
        "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "quantization": "Q4_K_M",
        "ram_required": "8GB+"
    },
    "gemma-2b": {
        "name": "Gemma (2B)",
        "description": "Google's Gemma 2B parameter model - lightweight and efficient",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "Low-Medium",
        "latency": "Low",
        "category": "local",
        "download_required": True,
        "model_id": "google/gemma-2b-it",
        "quantization": "None",
        "ram_required": "8GB+"
    },
    "gemma-2b-q4": {
        "name": "Gemma (2B) - 4-bit Quantized",
        "description": "Google's Gemma 2B model with 4-bit quantization - very low memory requirements",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "Low-Medium",
        "latency": "Low",
        "category": "local",
        "download_required": True,
        "model_id": "TheBloke/gemma-2b-it-GGUF",
        "quantization": "Q4_K_M",
        "ram_required": "4GB+"
    }
}

def get_available_llms() -> Dict[str, Dict[str, Any]]:
    """
    Get all available LLMs with their configurations.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of LLM configurations
    """
    return AVAILABLE_LLMS

def get_llm_info(llm_id: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific LLM.
    
    Args:
        llm_id (str): The LLM identifier
        
    Returns:
        Optional[Dict[str, Any]]: LLM configuration or None if not found
    """
    return AVAILABLE_LLMS.get(llm_id)

def check_api_key_availability(api_key_name: str) -> bool:
    """
    Check if an API key is available in the environment.
    
    Args:
        api_key_name (str): The name of the environment variable for the API key
        
    Returns:
        bool: True if the API key is available, False otherwise
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if the API key is available
    return os.environ.get(api_key_name) is not None

def is_model_downloaded(model_id: str, quantization: Optional[str] = None) -> bool:
    """
    Check if a Hugging Face model is already downloaded.
    
    Args:
        model_id (str): The Hugging Face model ID
        quantization (Optional[str]): Quantization level, if applicable
        
    Returns:
        bool: True if the model is downloaded, False otherwise
    """
    # This is a placeholder implementation
    # In a real implementation, this would check the local model cache
    # For now, we'll just return False to indicate models need to be downloaded
    return False

def download_model(model_id: str, quantization: Optional[str] = None) -> bool:
    """
    Download a Hugging Face model.
    
    Args:
        model_id (str): The Hugging Face model ID
        quantization (Optional[str]): Quantization level, if applicable
        
    Returns:
        bool: True if the download was successful, False otherwise
    """
    # This is a placeholder implementation
    # In a real implementation, this would download the model using the Hugging Face API
    # For now, we'll just return True to simulate a successful download
    return True

def get_llm_availability() -> Dict[str, bool]:
    """
    Check the availability of all LLMs.
    
    Returns:
        Dict[str, bool]: Dictionary mapping LLM IDs to their availability status
    """
    availability = {}
    
    for llm_id, llm_info in AVAILABLE_LLMS.items():
        if llm_info.get("requires_api_key", False):
            # Check if API key is available
            api_key_name = llm_info.get("api_key_name")
            availability[llm_id] = check_api_key_availability(api_key_name) if api_key_name else False
        elif llm_info.get("download_required", False):
            # Check if model is downloaded
            model_id = llm_info.get("model_id")
            quantization = llm_info.get("quantization")
            availability[llm_id] = is_model_downloaded(model_id, quantization) if model_id else False
        else:
            # Model is always available
            availability[llm_id] = True
    
    return availability