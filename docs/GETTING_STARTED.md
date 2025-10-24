# Getting Started Guide

Complete setup and configuration guide for the Innovation-Duplication project.

## 📋 Prerequisites

- Python 3.11+
- Azure OpenAI API access
- Git

## 🚀 Quick Setup (5 minutes)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.template .env
```

Fill in `.env`:

```bash
# Required for the pipeline
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15

# Optional – enables the Streamlit chatbot tab
AZURE_AI_SEARCH_KEY=
AZURE_AI_SEARCH_ENDPOINT=
AZURE_AI_SEARCH_INDEX_NAME=innovation-index
```

### 3. Verify connectivity

```bash
python tests/integration/test_azure_connection.py
```

Expected output:

```
✅ Chat Model (GPT-4o-mini)
✅ Embedding Model (text-embedding-3-small)
🎉 All tests passed!
```

If the script cannot find your deployment, confirm you used the **deployment name** (Azure portal → Model deployments), not only the model family.

### 4. Optional smoke checks

- Unit tests: `PYTHONPATH=src pytest tests/unit`
- Data sanity: ensure `data/dataframes/` and each `data/graph_docs_*` folder exists

## 🎯 First Run

### Option 1: Explore the raw data

```bash
jupyter notebook notebooks/introduction_data.ipynb
```

### Option 2: Execute the full pipeline

```bash
PYTHONPATH=src python -m innovation_platform.innovation_resolution
```

Helpful flags:

- `--skip-eval` to iterate faster
- `--skip-visualization` / `--skip-export` to stop after deduplication
- `--clustering-method graph_threshold --similarity-threshold 0.88` for graph-based clustering

### Option 3: Launch the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 and follow the on-screen prompts. The chatbot requires Azure AI Search credentials.

### Option 4: Serve generated artifacts

```bash
./start_server.sh 8000   # pass a different port if needed
```

The script verifies the `results/` folder and prints direct links to the HTML and PNG outputs.

## ⚙️ Configuration Details

### Azure OpenAI Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | Your API key | `2P5da...` |
| `AZURE_OPENAI_ENDPOINT` | Base URL (with trailing `/`) | `https://xxx.cognitiveservices.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | API version for chat | `2025-01-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name | `gpt-4o-mini` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment | `text-embedding-3-small` |
| `AZURE_OPENAI_EMBEDDING_API_VERSION` | API version for embeddings | `2023-05-15` |

**Important:** 
- Use **base URL only**, not the full API path
- ✅ Correct: `https://xxx.cognitiveservices.azure.com/`
- ❌ Wrong: `https://xxx.cognitiveservices.azure.com/openai/deployments/...`

### Getting API Keys

1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to your Azure OpenAI resource
3. Click **Keys and Endpoint**
4. Copy **Key 1** and **Endpoint**
5. Find your **Deployment names** in Model deployments

### Azure AI Search (Optional)

Only needed for the Streamlit app's Q&A chatbot feature. Core functionality works without it.

```bash
AZURE_AI_SEARCH_KEY=your-search-key
AZURE_AI_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_AI_SEARCH_INDEX_NAME=innovation-index
```

## 🔧 Command Line Options

```bash
# Standard run
PYTHONPATH=src python -m innovation_platform.innovation_resolution

# Fast run (skip evaluation)
PYTHONPATH=src python -m innovation_platform.innovation_resolution --skip-eval

# Custom cache location
PYTHONPATH=src python -m innovation_platform.innovation_resolution --cache-path "./cache/embeddings.json"

# Disable caching
PYTHONPATH=src python -m innovation_platform.innovation_resolution --no-cache
```

See **CLI_USAGE_GUIDE.md** for a full breakdown of flags, presets, and automation recipes.

**All options:**
- `--cache-type TYPE` - Cache type (default: embedding)
- `--cache-backend TYPE` - Backend (json or memory)
- `--cache-path PATH` - Cache file path
- `--no-cache` - Disable caching
- `--skip-eval` - Skip evaluation
- `--auto-label` - Auto-label for evaluation

## 🐛 Troubleshooting

### "API key not found"
**Solution:** Check `.env` exists and contains `AZURE_OPENAI_API_KEY`

### "Invalid endpoint"
**Causes:**
- Endpoint includes full path (use base URL only)
- Missing trailing slash

**Fix:**
```bash
# Wrong
AZURE_OPENAI_ENDPOINT=https://xxx.com/openai/deployments/gpt-4o-mini/chat/completions

# Correct
AZURE_OPENAI_ENDPOINT=https://xxx.cognitiveservices.azure.com/
```

### "Deployment not found"
**Check:**
- Deployment name matches exactly (case-sensitive)
- Deployment is active in Azure Portal
- Using correct Azure resource

### Connection test fails
```bash
# Run diagnostic
python tests/integration/test_azure_connection.py
```

Common issues:
- ❌ Invalid API key
- ❌ Wrong endpoint format
- ❌ Incorrect API version
- ❌ Deployment doesn't exist
- ❌ Network connectivity

## 🔄 Migration from JSON Config

If you have an existing `data/keys/azure_config.json`:

### Automatic (Recommended)
The config loader automatically uses `.env` if available, falls back to JSON.

### Manual Migration
1. Copy values from `azure_config.json` to `.env`
2. Test with `python tests/integration/test_azure_connection.py`
3. Keep JSON as backup until verified

**Old format:**
```json
{
  "gpt-4o-mini": {
    "api_key": "...",
    "api_base": "...",
    "api_version": "..."
  }
}
```

**New format:**
```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=...
```

### Code Changes
```python
# Old
import json
with open('data/keys/azure_config.json') as f:
    config = json.load(f)

# New (recommended)
from config import load_config
config = load_config()
```

## 📊 What the Analysis Does

When you run `PYTHONPATH=src python -m innovation_platform.innovation_resolution`:

1. ✅ Loads innovation data from multiple sources
2. ✅ Generates embeddings using Azure OpenAI
3. ✅ Detects duplicate innovations (semantic similarity)
4. ✅ Creates consolidated knowledge graph
5. ✅ Analyzes innovation network
6. ✅ Generates visualizations
7. ✅ Exports results to `results/` directory
8. ✅ Runs evaluation metrics

## 📁 Output Files

### results/
- `canonical_mapping.json` - Innovation ID mappings
- `consolidated_graph.json` - Complete knowledge graph
- `innovation_network_3d.html` - Interactive 3D visualization
- `innovation_stats.png` - Statistics charts

### evaluation/
- `evaluation_results.json` - Quality metrics
- `pred_entities.json` - Predicted entities
- `pred_relations.json` - Predicted relations

## 🔒 Security Best Practices

### ✅ DO:
- Keep `.env` in `.gitignore`
- Use separate keys for dev/prod
- Rotate keys regularly
- Never commit credentials

### ❌ DON'T:
- Commit `.env` to git
- Share API keys
- Hardcode credentials
- Push keys to public repos

## 💡 Tips

1. **Start simple** - Run with default options first
2. **Use caching** - Saves time and API costs
3. **Test often** - Run connection tests after config changes
4. **Check logs** - Look for errors in console output
5. **Read docs** - Check [DEVELOPMENT.md](DEVELOPMENT.md) for advanced usage

## 🆘 Need Help?

- **Project structure** → [DEVELOPMENT.md](DEVELOPMENT.md)
- **Main README** → [../README.md](../README.md)
- **Test suite** → [../tests/README.md](../tests/README.md)

---

*Last updated: 2025-10-16*
