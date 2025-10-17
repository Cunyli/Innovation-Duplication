# Configuration Module

This module handles all configuration management for the Innovation-Duplication project.

## Files

### `config_loader.py`
Unified configuration loader that supports multiple configuration sources:
- `.env` files (preferred)
- `azure_config.json` (legacy)
- `.streamlit/secrets.toml` (Streamlit apps)

**Usage:**
```python
from config import load_config, get_model_config, initialize_llm_from_env

# Load configuration
config = load_config()

# Get specific model config
model_config = get_model_config('gpt-4o-mini')

# Initialize LLM directly
llm = initialize_llm_from_env('gpt-4o-mini')
```

### `generate_config_from_toml.py`
Converts `.streamlit/secrets.toml` to `data/keys/azure_config.json` format.

Used by devcontainer for Streamlit deployments.

**Usage:**
```bash
python config/generate_config_from_toml.py
```

## Configuration Priority

1. **`.env` file** (if `prefer_env=True`, default)
2. **`azure_config.json`** (fallback)
3. **Error** if neither found

## API Reference

### `load_config(config_path=None, prefer_env=True)`
Load configuration from available sources.

**Parameters:**
- `config_path` (str, optional): Path to JSON config file
- `prefer_env` (bool): Whether to prefer .env over JSON

**Returns:** Configuration dictionary

### `load_config_from_env()`
Load configuration from environment variables.

**Returns:** Configuration dictionary compatible with azure_config.json format

### `load_config_from_json(config_path)`
Load configuration from JSON file.

**Parameters:**
- `config_path` (str): Path to azure_config.json

**Returns:** Configuration dictionary

### `get_model_config(model_name='gpt-4o-mini', config=None)`
Get configuration for a specific model.

**Parameters:**
- `model_name` (str): Model deployment name
- `config` (dict, optional): Pre-loaded config

**Returns:** Model-specific configuration dictionary

### `initialize_llm_from_env(deployment_model='gpt-4o-mini')`
Initialize Azure OpenAI LLM using environment variables.

**Parameters:**
- `deployment_model` (str): Model deployment name

**Returns:** AzureChatOpenAI instance

## Environment Variables

See [../docs/CONFIGURATION.md](../docs/CONFIGURATION.md) for complete list.

### Required:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

### Optional:
- `AZURE_AI_SEARCH_KEY`
- `AZURE_AI_SEARCH_ENDPOINT`
- `AZURE_AI_SEARCH_INDEX_NAME`

## Examples

### Basic Usage
```python
from config import load_config

config = load_config()
print(config['gpt-4o-mini']['api_key'])
```

### Using with LangChain
```python
from config import initialize_llm_from_env

llm = initialize_llm_from_env('gpt-4o-mini')
response = llm.invoke("Hello, world!")
print(response.content)
```

### Force Configuration Source
```python
from config import load_config

# Force .env
config = load_config(prefer_env=True)

# Force JSON
config = load_config(prefer_env=False, config_path='data/keys/azure_config.json')
```

## Testing

Test configuration:
```bash
python config/config_loader.py
```

Test API connectivity:
```bash
python tests/test_azure_connection.py
```

## Migration

See [../docs/MIGRATION_GUIDE.md](../docs/MIGRATION_GUIDE.md) for migrating from JSON to .env.
