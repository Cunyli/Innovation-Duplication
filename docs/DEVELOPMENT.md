# Development Guide

Developer guide for understanding and extending the Innovation-Duplication project.

## 📁 Project Structure

```
Innovation-Duplication/
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── config_loader.py        # Unified config loader
│   ├── generate_config_from_toml.py
│   └── README.md
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_azure_connection.py
│   ├── test_cluster.py
│   └── README.md
│
├── docs/                       # Documentation
│   ├── README.md               # This file
│   ├── GETTING_STARTED.md      # Setup & configuration
│   └── DEVELOPMENT.md          # Development guide
│
├── data/                       # Data files (git-ignored)
│   ├── dataframes/             # Source data
│   ├── entity_glossary/        # Name resolution
│   ├── graph_docs/             # Extracted relationships
│   └── keys/                   # API configs (git-ignored)
│
├── evaluation/                 # Evaluation files
│   ├── *_template.json/csv     # Templates (tracked)
│   └── *.json/csv              # Results (git-ignored)
│
├── results/                    # Output results
│   ├── *.json                  # Data files
│   ├── *.html                  # Interactive visualizations
│   └── *.png                   # Static visualizations
│
├── utils/                      # Utility modules
│   └── cluster/                # Clustering algorithms
│
├── scripts/                    # Utility scripts (future)
│
├── .env                        # Environment config (git-ignored)
├── .env.template               # Config template
├── app.py                      # Streamlit web app
├── innovation_resolution.py    # Main analysis script
├── innovation_utils.py         # Utility functions
├── evaluation.py               # Evaluation module
├── local_entity_processing.py  # Data models
├── vis.py                      # Visualization utilities
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

## 🔧 Core Components

### Main Application Files

#### `innovation_resolution.py`
**Purpose:** Main analysis pipeline  
**Usage:** `python innovation_resolution.py [options]`

**What it does:**
1. Loads innovation data from multiple sources
2. Resolves duplicate innovations using embeddings
3. Creates consolidated knowledge graph
4. Generates visualizations
5. Exports results
6. Runs evaluation

#### `app.py`
**Purpose:** Streamlit web interface  
**Usage:** `streamlit run app.py`

**Features:**
- Interactive data exploration
- Q&A chatbot (requires Azure AI Search)
- Visualization tools

#### `evaluation.py`
**Purpose:** Quality assessment module

**Metrics:**
- Entity and relation evaluation
- Consistency checking
- QA validation

### Configuration Module (`config/`)

#### `config_loader.py`
Unified configuration loader supporting:
- `.env` files (recommended)
- `azure_config.json` (legacy)
- Automatic fallback

**Usage:**
```python
from config import load_config, initialize_llm_from_env

# Load config
config = load_config()

# Initialize LLM
llm = initialize_llm_from_env('gpt-4o-mini')
```

#### `generate_config_from_toml.py`
Converts `.streamlit/secrets.toml` to JSON format.

### Test Suite (`tests/`)

#### `test_azure_connection.py`
Tests Azure OpenAI API connectivity for chat and embedding models.

```bash
python tests/test_azure_connection.py
```

#### `test_cluster.py`
Tests clustering algorithms used for deduplication.

```bash
python tests/test_cluster.py
```

## 🔄 Data Flow

```
Source Data (graph_docs/)
    ↓
Load & Preprocess
    ↓
Generate Embeddings (Azure OpenAI)
    ↓
Detect Duplicates (Clustering)
    ↓
Consolidate Graph
    ↓
Analyze Network
    ↓
Generate Visualizations
    ↓
Export Results (results/)
    ↓
Run Evaluation (evaluation/)
```

## 🎨 Key Algorithms

### 1. Innovation Deduplication
- Uses semantic embeddings
- Clustering with HDBSCAN
- Similarity threshold-based merging

### 2. Knowledge Graph Construction
- Combines data from multiple sources
- Resolves entity names
- Preserves source attribution

### 3. Network Analysis
- Identifies key organizations
- Finds collaboration patterns
- Calculates network metrics

## 🔌 API Usage

### Loading Configuration

```python
from config import load_config, get_model_config

# Load all config
config = load_config()

# Get specific model config
model_config = get_model_config('gpt-4o-mini')

# Access values
api_key = model_config['api_key']
endpoint = model_config['api_base']
```

### Initializing Models

```python
from config import initialize_llm_from_env
from langchain_openai import AzureOpenAIEmbeddings

# Chat model
llm = initialize_llm_from_env('gpt-4o-mini')
response = llm.invoke("Hello!")

# Embedding model
from config import load_config
config = load_config()
model_config = config['gpt-4o-mini']

embeddings = AzureOpenAIEmbeddings(
    model=model_config['emb_deployment'],
    api_key=model_config['api_key'],
    azure_endpoint=model_config['api_base'],
    api_version=model_config.get('emb_api_version', model_config['api_version'])
)
```

### Custom Processing

```python
from innovation_utils import get_embedding
from local_entity_processing import GraphDocument

# Generate embedding
text = "Some innovation description"
embedding = get_embedding(text, model)

# Load graph documents
with open('data/graph_docs/file.json') as f:
    data = json.load(f)
    doc = GraphDocument(**data)
```

## 🛠️ Development Workflow

### 1. Setup Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your keys

# Test setup
python tests/test_azure_connection.py
```

### 2. Make Changes

Edit the relevant files:
- **Core logic:** `innovation_resolution.py`
- **Utilities:** `innovation_utils.py`
- **Data models:** `local_entity_processing.py`
- **Visualization:** `vis.py`
- **Web app:** `app.py`

### 3. Test Changes

```bash
# Test configuration
python tests/test_azure_connection.py

# Run with skip-eval for faster iteration
python innovation_resolution.py --skip-eval

# Test specific features
python -c "from innovation_utils import function_name; test_code()"
```

### 4. Update Documentation

If you add features:
- [ ] Update relevant docstrings
- [ ] Update README.md if needed
- [ ] Add examples to docs/

## 📦 Adding New Features

### Adding a New Clustering Algorithm

1. Add to `utils/cluster/cluster_algorithms.py`
2. Update imports in main script
3. Add tests in `tests/test_cluster.py`
4. Document in code and README

### Adding a New Data Source

1. Create parser in `local_entity_processing.py`
2. Update data loading in `innovation_resolution.py`
3. Add to data/ directory structure
4. Document format and usage

### Adding New Evaluation Metrics

1. Extend `evaluation.py`
2. Update evaluation templates in `evaluation/`
3. Add to results output
4. Document metrics

## 🔍 Code Organization Principles

### 1. Separation of Concerns
- **Configuration:** `config/`
- **Core logic:** Main scripts
- **Utilities:** `utils/`, `*_utils.py`
- **Data models:** `local_entity_processing.py`
- **Tests:** `tests/`

### 2. Module Independence
- Each module should work standalone
- Clear interfaces between modules
- Minimal cross-dependencies

### 3. Configuration Management
- Use `.env` for local development
- Use `azure_config.json` for legacy compatibility
- Never hardcode credentials

## 🔐 Security Guidelines

### Environment Variables
```bash
# ✅ Good
from config import load_config
config = load_config()

# ❌ Bad
api_key = "hardcoded-key"
```

### Git Ignore
Ensure these are in `.gitignore`:
```
.env
.env.local
data/keys/
*.cache.json
```

### API Keys
- Rotate regularly
- Use separate keys for dev/prod
- Never commit to repository
- Use Azure Key Vault in production

## 🧪 Testing Strategy

### Unit Tests
Test individual functions and components:
```python
def test_function():
    result = function_to_test(input)
    assert result == expected_output
```

### Integration Tests
Test API connectivity and end-to-end flows:
```bash
python tests/test_azure_connection.py
```

### Manual Testing
```bash
# Quick test run
python innovation_resolution.py --skip-eval --no-cache

# Full test run
python innovation_resolution.py
```

## 📝 Code Style

### Python Conventions
- Follow PEP 8
- Use type hints where appropriate
- Write descriptive docstrings
- Keep functions focused and small

### Documentation
- Update README.md for user-facing changes
- Update docs/ for architectural changes
- Add inline comments for complex logic
- Include usage examples

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production (Streamlit Cloud)
1. Push to GitHub
2. Configure `.streamlit/secrets.toml` in Streamlit Cloud
3. Deploy from Streamlit dashboard

### Docker (Future)
```dockerfile
FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## 📚 Resources

### Internal
- [Getting Started](GETTING_STARTED.md) - Setup guide
- [Main README](../README.md) - Project overview
- [Test README](../tests/README.md) - Test documentation
- [Config README](../config/README.md) - Configuration API

### External
- [Azure OpenAI Docs](https://learn.microsoft.com/azure/cognitive-services/openai/)
- [LangChain Docs](https://python.langchain.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [NetworkX Docs](https://networkx.org/)

## 🐛 Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Configuration
```python
from config import load_config
import json
config = load_config()
print(json.dumps(config, indent=2))
```

### Test API Calls
```python
python tests/test_azure_connection.py
```

### Inspect Data
```python
import json
with open('data/graph_docs/file.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
```

---

*Last updated: 2025-10-16*
