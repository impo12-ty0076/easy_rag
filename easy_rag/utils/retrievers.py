"""
Utility functions for retriever configuration and management.
"""
from typing import Dict, List, Any, Optional, Tuple

# Define available reranking LLMs
RERANKING_LLMS = {
    "none": {
        "name": "No Reranking",
        "description": "Don't use reranking",
        "requires_api_key": False,
        "icon": "times-circle",
        "performance": "N/A",
        "latency": "None"
    },
    "cohere-rerank": {
        "name": "Cohere Rerank",
        "description": "Cohere's specialized reranking model for improved relevance",
        "requires_api_key": True,
        "api_key_name": "COHERE_API_KEY",
        "icon": "cloud",
        "performance": "High",
        "latency": "Medium"
    },
    "bge-reranker-base": {
        "name": "BGE Reranker Base",
        "description": "Local BGE reranking model (base size) - good balance of performance and speed",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "Medium",
        "latency": "Low"
    },
    "bge-reranker-large": {
        "name": "BGE Reranker Large",
        "description": "Local BGE reranking model (large size) - higher quality but slower",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "High",
        "latency": "Medium"
    },
    "cross-encoder-ms-marco": {
        "name": "Cross-Encoder MS MARCO",
        "description": "Local cross-encoder model trained on MS MARCO dataset",
        "requires_api_key": False,
        "icon": "microchip",
        "performance": "Medium",
        "latency": "Medium"
    },
    "openai-rerank": {
        "name": "OpenAI Rerank",
        "description": "OpenAI's reranking capabilities (requires API key)",
        "requires_api_key": True,
        "api_key_name": "OPENAI_API_KEY",
        "icon": "cloud",
        "performance": "Very High",
        "latency": "Medium"
    }
}

# Define available retriever types
RETRIEVER_TYPES = {
    "similarity": {
        "name": "Similarity Search",
        "description": "Basic similarity search using vector embeddings",
        "icon": "search",
        "category": "basic",
        "parameters": {
            "k": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "description": "Number of documents to retrieve"
            },
            "score_threshold": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum similarity score threshold (0 = no threshold)",
                "required": False
            }
        }
    },
    "mmr": {
        "name": "Maximal Marginal Relevance",
        "description": "Balances relevance with diversity in search results",
        "icon": "shuffle",
        "category": "advanced",
        "parameters": {
            "k": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "description": "Number of documents to retrieve"
            },
            "fetch_k": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 50,
                "description": "Number of documents to fetch before filtering"
            },
            "lambda_mult": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Diversity vs relevance factor (0 = max diversity, 1 = max relevance)"
            }
        }
    },
    "contextual_compression": {
        "name": "Contextual Compression",
        "description": "Compresses retrieved documents to focus on relevant parts",
        "icon": "compress",
        "category": "advanced",
        "parameters": {
            "k": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "description": "Number of documents to retrieve"
            },
            "compression_ratio": {
                "type": "float",
                "default": 0.7,
                "min": 0.1,
                "max": 1.0,
                "description": "Target compression ratio (lower means more compression)"
            }
        }
    },
    "self_query": {
        "name": "Self Query",
        "description": "Automatically extracts filters from the query",
        "icon": "filter",
        "category": "advanced",
        "parameters": {
            "k": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "description": "Number of documents to retrieve"
            },
            "structured_query_mode": {
                "type": "select",
                "default": "simple",
                "options": ["simple", "advanced"],
                "description": "Query parsing complexity"
            }
        }
    },
    "multi_query": {
        "name": "Multi Query",
        "description": "Generates multiple query variations to improve retrieval",
        "icon": "list",
        "category": "advanced",
        "parameters": {
            "k": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "description": "Number of documents to retrieve per query"
            },
            "num_queries": {
                "type": "int",
                "default": 3,
                "min": 2,
                "max": 5,
                "description": "Number of query variations to generate"
            },
            "query_generation_mode": {
                "type": "select",
                "default": "standard",
                "options": ["standard", "detailed", "concise"],
                "description": "Query generation approach"
            }
        }
    },
    "hybrid": {
        "name": "Hybrid Search",
        "description": "Combines keyword and semantic search for better results",
        "icon": "layers",
        "category": "advanced",
        "parameters": {
            "k": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "description": "Number of documents to retrieve"
            },
            "alpha": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Weight between keyword (0) and semantic (1) search"
            },
            "use_splade": {
                "type": "boolean",
                "default": False,
                "description": "Use SPLADE for sparse retrieval (requires additional model)"
            }
        }
    },
    "ensemble": {
        "name": "Ensemble Retriever",
        "description": "Combines results from multiple retrievers",
        "icon": "people-group",
        "category": "experimental",
        "parameters": {
            "k": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "description": "Number of documents to retrieve"
            },
            "retriever_weights": {
                "type": "text",
                "default": "similarity:0.5,mmr:0.5",
                "description": "Comma-separated list of retriever:weight pairs"
            },
            "merge_mode": {
                "type": "select",
                "default": "reciprocal_rank_fusion",
                "options": ["reciprocal_rank_fusion", "simple_weight", "max_score"],
                "description": "Method to merge results from different retrievers"
            }
        }
    }
}

def get_retriever_types() -> Dict[str, Dict[str, Any]]:
    """
    Get all available retriever types with their configurations.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of retriever types
    """
    return RETRIEVER_TYPES

def get_retriever_type(retriever_type: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific retriever type.
    
    Args:
        retriever_type (str): The retriever type identifier
        
    Returns:
        Optional[Dict[str, Any]]: Retriever configuration or None if not found
    """
    return RETRIEVER_TYPES.get(retriever_type)

def validate_retriever_parameters(retriever_type: str, parameters: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate parameters for a specific retriever type.
    
    Args:
        retriever_type (str): The retriever type identifier
        parameters (Dict[str, Any]): Parameters to validate
        
    Returns:
        Dict[str, str]: Dictionary of validation errors (empty if valid)
    """
    errors = {}
    retriever_config = get_retriever_type(retriever_type)
    
    if not retriever_config:
        errors["retriever_type"] = f"Unknown retriever type: {retriever_type}"
        return errors
    
    for param_name, param_config in retriever_config["parameters"].items():
        # Skip validation for optional parameters that are not provided
        if param_name not in parameters and param_config.get("required", True) is False:
            continue
            
        # Check if required parameter is missing
        if param_name not in parameters and param_config.get("required", True):
            errors[param_name] = "This field is required"
            continue
            
        # Skip empty optional parameters
        if param_name not in parameters:
            continue
            
        param_value = parameters[param_name]
        
        # Skip empty values for optional parameters
        if not param_value and param_config.get("required", True) is False:
            continue
            
        # Type validation
        if param_config["type"] == "int":
            try:
                param_value = int(param_value)
                if param_value < param_config["min"] or param_value > param_config["max"]:
                    errors[param_name] = f"Value must be between {param_config['min']} and {param_config['max']}"
            except (ValueError, TypeError):
                errors[param_name] = "Value must be an integer"
                
        elif param_config["type"] == "float":
            try:
                param_value = float(param_value)
                if param_value < param_config["min"] or param_value > param_config["max"]:
                    errors[param_name] = f"Value must be between {param_config['min']} and {param_config['max']}"
            except (ValueError, TypeError):
                errors[param_name] = "Value must be a number"
                
        elif param_config["type"] == "select":
            if param_value not in param_config["options"]:
                errors[param_name] = f"Value must be one of: {', '.join(param_config['options'])}"
                
        elif param_config["type"] == "boolean":
            # Convert string representations to boolean
            if isinstance(param_value, str):
                if param_value.lower() in ('true', 'yes', '1', 'on'):
                    parameters[param_name] = True
                elif param_value.lower() in ('false', 'no', '0', 'off'):
                    parameters[param_name] = False
                else:
                    errors[param_name] = "Value must be a boolean (true/false)"
            elif not isinstance(param_value, bool):
                errors[param_name] = "Value must be a boolean"
                
        elif param_config["type"] == "text":
            # Text validation can be extended as needed
            if not isinstance(param_value, str):
                errors[param_name] = "Value must be text"
    
    return errors

def get_default_parameters(retriever_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific retriever type.
    
    Args:
        retriever_type (str): The retriever type identifier
        
    Returns:
        Dict[str, Any]: Dictionary of default parameters
    """
    defaults = {}
    retriever_config = get_retriever_type(retriever_type)
    
    if not retriever_config:
        return defaults
    
    for param_name, param_config in retriever_config["parameters"].items():
        defaults[param_name] = param_config["default"]
    
    return defaults

def get_reranking_llms() -> Dict[str, Dict[str, Any]]:
    """
    Get all available reranking LLMs with their configurations.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of reranking LLMs
    """
    return RERANKING_LLMS

def get_reranking_llm(llm_id: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific reranking LLM.
    
    Args:
        llm_id (str): The LLM identifier
        
    Returns:
        Optional[Dict[str, Any]]: LLM configuration or None if not found
    """
    return RERANKING_LLMS.get(llm_id)

def check_api_key_availability(api_key_name: str) -> bool:
    """
    Check if an API key is available in the environment.
    
    Args:
        api_key_name (str): The name of the environment variable for the API key
        
    Returns:
        bool: True if the API key is available, False otherwise
    """
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if the API key is available
    return os.environ.get(api_key_name) is not None

def validate_advanced_retrieval_options(options: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate advanced retrieval options.
    
    Args:
        options (Dict[str, Any]): Advanced retrieval options to validate
        
    Returns:
        Dict[str, str]: Dictionary of validation errors (empty if valid)
    """
    errors = {}
    
    # Validate reranking LLM
    if 'reranking_llm' in options:
        llm_id = options['reranking_llm']
        llm_config = get_reranking_llm(llm_id)
        
        if not llm_config:
            errors['reranking_llm'] = f"Unknown reranking LLM: {llm_id}"
        elif llm_config.get('requires_api_key', False):
            api_key_name = llm_config.get('api_key_name')
            if api_key_name and not check_api_key_availability(api_key_name):
                errors['reranking_llm'] = f"API key not found for {llm_config['name']}. Please add {api_key_name} to your .env file."
    else:
        # Default to "none" if not provided
        options['reranking_llm'] = 'none'
    
    # Validate chunk count
    if 'chunk_count' in options:
        try:
            chunk_count = int(options['chunk_count'])
            if chunk_count < 1 or chunk_count > 20:
                errors['chunk_count'] = "Chunk count must be between 1 and 20"
            else:
                # Convert to int to ensure proper storage
                options['chunk_count'] = chunk_count
        except (ValueError, TypeError):
            errors['chunk_count'] = "Chunk count must be an integer"
    else:
        # Default to 4 if not provided
        options['chunk_count'] = 4
    
    # Validate hybrid search settings
    if 'use_hybrid_search' in options:
        # Convert string representations to boolean
        if isinstance(options['use_hybrid_search'], str):
            if options['use_hybrid_search'].lower() in ('true', 'yes', '1', 'on'):
                options['use_hybrid_search'] = True
            else:
                options['use_hybrid_search'] = False
    else:
        # Default to False if not provided
        options['use_hybrid_search'] = False
    
    # Only validate hybrid_alpha if hybrid search is enabled
    if options['use_hybrid_search']:
        if 'hybrid_alpha' in options:
            try:
                hybrid_alpha = float(options['hybrid_alpha'])
                if hybrid_alpha < 0.0 or hybrid_alpha > 1.0:
                    errors['hybrid_alpha'] = "Hybrid alpha must be between 0.0 and 1.0"
                else:
                    # Convert to float to ensure proper storage
                    options['hybrid_alpha'] = hybrid_alpha
            except (ValueError, TypeError):
                errors['hybrid_alpha'] = "Hybrid alpha must be a number"
        else:
            # Default to 0.5 if not provided
            options['hybrid_alpha'] = 0.5
    
    return errors