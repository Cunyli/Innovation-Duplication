# Development Guide

This guide explains how the project is organised, how the pipeline works, and what to touch when you extend it.

## 📁 Repository Layout

```
Innovation-Duplication/
├── app.py                    # Streamlit front-end
├── notebooks/                # Exploratory notebooks
├── scripts/                  # Helper scripts (analytics, demos)
├── src/
│   └── innovation_platform/
│       ├── __init__.py
│       ├── config/           # Config loader + generators
│       ├── core/             # Cache backends
│       ├── data_pipeline/    # Loaders, processors, feature builders
│       ├── utils/            # Shared helpers (clustering)
│       ├── evaluation.py     # Evaluation orchestration
│       ├── innovation_resolution.py
│       ├── innovation_utils.py
│       ├── local_entity_processing.py
│       └── vis.py
├── data/                     # Input assets (gitignored)
│   ├── dataframes/
│   ├── graph_docs_*/
│   ├── entity_glossary/
│   └── keys/
├── results/                  # Generated artifacts (json/png/html)
├── evaluation/               # Label templates & sampled outputs
├── tests/                    # Unit & integration tests
└── docs/                     # Documentation bundle
```

### Key packages inside `src/innovation_platform`

| Module | Purpose | Highlights |
|--------|---------|-----------|
| `config` | Normalises config from `.env` or legacy JSON | `load_config`, `initialize_llm_from_env` |
| `core` | Cache abstractions for embeddings | JSON + in-memory backends |
| `data_pipeline.loaders` | Reads graph pickle files & node lookups | `GraphDocumentLoader`, `NodeMapper` |
| `data_pipeline.processors` | Converts raw docs into structured relationships, builds features, runs clustering | `DataSourceProcessor`, `InnovationFeatureBuilder`, `ClusteringStrategyFactory` |
| `utils.cluster` | Concrete clustering algorithms + graph-based helpers | `cluster_with_stats`, `graph_threshold_clustering` |
| `vis` | Tufte-inspired plots + 3D network renderer | Saves into `results/` |
| `evaluation` | Wraps consistency/entity/relation/QA checks | `run_all_evaluations` |

## 🔄 Pipeline Overview

`innovation_platform.innovation_resolution` wires the end-to-end workflow:

1. **Load** CSVs and graph pickles (`load_and_combine_data`).
2. **Feature** each innovation (concatenate descriptions, relationships, sources).
3. **Embed** innovations and fetch cached vectors when possible (`EmbeddingManager`).
4. **Cluster** duplicates (default HDBSCAN; graph or flat clustering supported).
5. **Consolidate** a knowledge graph that merges nodes and relationships.
6. **Analyse & Visualise** centralities, top organisations, and network snapshots (`vis.py`).
7. **Export** JSON summaries/PNGs/HTML (`export_results`).
8. **Evaluate** predictions against gold templates (`run_all_evaluations`) — optional.

### Semantic query layer
- `InnovationQueryEngine` builds textual representations for each canonical innovation, embeds them (with caching), and supports cosine-similarity search.
- Use `query_innovations(consolidated_graph, "my query")` for a quick lookup or instantiate the engine directly for repeated queries.

### Orchestration with `pipeline_runner`
Run `PYTHONPATH=src python -m innovation_platform.pipeline_runner --resume` to execute the full workflow. The runner:

- Detects existing artefacts (consolidated graph, canonical mapping, embedding caches) and skips redundant steps unless `--force` is passed.
- Provides flags such as `--sample-query` to warm up the semantic search and `--with-eval` to trigger the evaluation suite.
- Records the last run in `.pipeline_state.json`, making it easy to track when artefacts were refreshed.

Each step can be toggled via the CLI `--steps` argument or dedicated skip-flags. The defaults execute everything except the Streamlit front-end.

## 📦 Data & Artifacts

- **Inputs (`data/`)** — keep raw CSVs, processed graph pickles, and API keys.
- **Intermediate cache** — embedding cache defaults to `embedding_vectors.json` in the repo root; override with `--cache-path`.
- **Outputs (`results/`)** — canonical mapping, consolidated graph, PNG/HTML visualisations, stats.
- **Evaluation (`evaluation/`)** — templates (`*_template.*`) remain in git; generated comparisons land alongside them.

Version large inputs externally (e.g. blob storage) and keep this repo focused on code + lightweight artefacts.

## 🧪 Testing & Quality

| Scope | Command | Notes |
|-------|---------|-------|
| Unit smoke tests | `PYTHONPATH=src pytest tests/unit` | No external dependencies |
| Integration tests | `PYTHONPATH=src pytest tests/integration` | Touches data + Azure creds |
| Azure connectivity | `python tests/integration/test_azure_connection.py` | Validates chat + embedding deployments |

When adding modules, mirror the layout under `tests/unit` or `tests/integration` to keep coverage obvious.

## 🛠️ Development Tips

- **Relative imports:** always import via `innovation_platform.<module>` to keep packaging clean.
- **Pickle compatibility:** existing pickles expect the `local_entity_processing` module; the shim at repo root maps to the new package — do not remove it.
- **Configuration defaults:** expose new settings through `.env` and surface them in `config_loader.py` first; reference them via `load_config()`.
- **CLI additions:** document any new arguments in `CLI_USAGE_GUIDE.md` and provide a shorthand example.
- **Visualisations:** write to `results/` (or the user-specified `--output-dir`) and avoid hard-coded paths elsewhere.

## 🤝 Contributing

1. Branch from `main` and keep PRs scoped by pipeline stage (e.g. loaders vs clustering).
2. Update relevant documentation — at minimum, reference new behaviour in `CLI_USAGE_GUIDE.md` or `ANALYSIS_RESULTS_CHEATSHEET.md`.
3. Run unit tests and, when feasible, the integration suite.
4. Attach sample outputs or screenshots for visual changes.

Need deeper algorithmic insight? See `TECHNICAL_DETAILS.md` for scoring heuristics, graph metrics, and rationale for the current clustering defaults.

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
python tests/integration/test_azure_connection.py
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
python tests/integration/test_azure_connection.py

# Run with skip-eval for faster iteration
PYTHONPATH=src python -m innovation_platform.innovation_resolution --skip-eval

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
python tests/integration/test_azure_connection.py
```

### Manual Testing
```bash
# Quick test run
PYTHONPATH=src python -m innovation_platform.innovation_resolution --skip-eval --no-cache

# Full test run
PYTHONPATH=src python -m innovation_platform.innovation_resolution
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
python tests/integration/test_azure_connection.py
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
