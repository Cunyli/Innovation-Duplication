"""
Configuration loader utility for migrating from azure_config.json to .env
This module provides backward-compatible configuration loading.
"""
import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_config_from_env() -> Dict[str, Any]:
    """
    Load configuration from environment variables (.env file)
    Returns a structure compatible with the old azure_config.json format
    """
    # Build the configuration dictionary matching azure_config.json structure
    config = {
        "gpt-4o-mini": {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "api_base": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            "eval_deployment": os.getenv("AZURE_OPENAI_EVAL_DEPLOYMENT", "gpt-4o-mini"),
            "emb_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
            "emb_api_version": os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"))
        },
        "azure-ai-search": {
            "api_key": os.getenv("AZURE_AI_SEARCH_KEY", ""),
            "azure_endpoint": os.getenv("AZURE_AI_SEARCH_ENDPOINT", ""),
            "index_name": os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "innovation-index")
        },
        "INPUT": {
            "first_order_k": int(os.getenv("INPUT_FIRST_ORDER_K", "10")),
            "second_order_k": int(os.getenv("INPUT_SECOND_ORDER_K", "15")),
            "temperature": float(os.getenv("INPUT_TEMPERATURE", "0.01")),
            "validity_threshold": float(os.getenv("INPUT_VALIDITY_THRESHOLD", "0.4")),
            "aggregation_threshold": float(os.getenv("INPUT_AGGREGATION_THRESHOLD", "0.5"))
        }
    }
    
    return config


def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file (legacy method)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def load_config(config_path: Optional[str] = None, prefer_env: bool = True) -> Dict[str, Any]:
    """
    Load configuration with fallback logic
    
    Args:
        config_path: Path to azure_config.json (optional)
        prefer_env: If True, prefer .env over JSON file
    
    Returns:
        Configuration dictionary
    """
    # If prefer_env is True and we have env vars, use them
    if prefer_env and os.getenv("AZURE_OPENAI_API_KEY"):
        print("✅ Loading configuration from .env file")
        return load_config_from_env()
    
    # Otherwise, try to load from JSON
    if config_path is None:
        config_path = os.path.join('data', 'keys', 'azure_config.json')
    
    if os.path.exists(config_path):
        print(f"✅ Loading configuration from {config_path}")
        return load_config_from_json(config_path)
    
    # Final fallback: try .env
    if os.getenv("AZURE_OPENAI_API_KEY"):
        print("⚠️ JSON config not found, falling back to .env")
        return load_config_from_env()
    
    raise FileNotFoundError(
        "No valid configuration found. Please set up either:\n"
        "1. .env file with AZURE_OPENAI_API_KEY, or\n"
        f"2. {config_path} file"
    )


def get_model_config(model_name: str = "gpt-4o-mini", config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific model
    
    Args:
        model_name: Model deployment name
        config: Pre-loaded config dict (optional)
    
    Returns:
        Model-specific configuration
    """
    if config is None:
        config = load_config()
    
    if model_name not in config:
        # If exact model not found, try to use the default structure
        default_model = "gpt-4o-mini"
        if default_model in config:
            print(f"⚠️ Model '{model_name}' not found, using '{default_model}' config")
            return config[default_model]
        raise KeyError(f"Model '{model_name}' not found in configuration")
    
    return config[model_name]


# For backward compatibility
def initialize_llm_from_env(deployment_model: str = "gpt-4o-mini"):
    """
    Initialize Azure OpenAI LLM using environment variables
    This is a drop-in replacement for the old initialize_llm function
    """
    from langchain_openai import AzureChatOpenAI
    
    config = load_config()
    model_config = get_model_config(deployment_model, config)
    
    return AzureChatOpenAI(
        model=deployment_model,
        api_key=model_config['api_key'],
        azure_endpoint=model_config['api_base'],
        api_version=model_config['api_version']
    )


if __name__ == "__main__":
    # Test the configuration loader
    print("Testing configuration loader...")
    try:
        config = load_config()
        print("\n📋 Loaded configuration:")
        print(json.dumps(config, indent=2))
        print("\n✅ Configuration loaded successfully!")
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}")
