# VTT Innovation Resolution

[![YouTube](https://img.shields.io/badge/YouTube-Demo-red)](https://www.youtube.com/watch?v=yKNr22bu9Yc)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://innovation-duplication.streamlit.app)

This project addresses the challenge of identifying and consolidating innovation disclosures from VTT's collaboration partnerships.

## 🚀 Quick Start

**New to this project?** Start here:

1. **[Getting Started Guide](docs/GETTING_STARTED.md)** - Complete setup in 5 minutes
2. **[Development Guide](docs/DEVELOPMENT.md)** - Understand the codebase

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Getting Started](docs/GETTING_STARTED.md)** | Setup, configuration & troubleshooting |
| **[Development](docs/DEVELOPMENT.md)** | Project structure & development guide |
| **[Clustering Guide](docs/CLUSTERING_GUIDE.md)** | Clustering methods & usage guide |
| **[Technical Details](docs/TECHNICAL_DETAILS.md)** | Deep dive into implementation |
| **[Data Pipeline Refactoring](docs/DATA_PIPELINE_REFACTORING_GUIDE.md)** | Architecture & design patterns |
| **[Documentation Index](docs/README.md)** | Full documentation overview |

## Challenge Description

The challenge focuses on two main tasks:

1. **Innovation Resolution**: Identify when different sources are discussing the same innovation by analyzing associated organizations, descriptions, and source text.

2. **Innovation Relationship**: Create a representation that aggregates information describing innovations without losing source attribution.

## Data Overview

The dataset contains two main sources:
- Company websites mentioning VTT collaborations
- VTT's website mentioning company collaborations

The data has been preprocessed into structured graph documents containing:
- Nodes (Organizations and Innovations)
- Relationships (DEVELOPED_BY and COLLABORATION)

## 📁 Project Structure

```
Innovation-Duplication/
├── config/                     # ⭐ Configuration management
│   ├── config_loader.py        # Unified config loader (.env + JSON)
│   └── generate_config_from_toml.py
├── tests/                      # ⭐ Test suite
│   ├── test_azure_connection.py
│   └── test_cluster.py         # Clustering algorithm tests
├── docs/                       # ⭐ Documentation
│   ├── GETTING_STARTED.md      # Quick setup guide
│   ├── DEVELOPMENT.md          # Development guide
│   ├── CLUSTERING.md           # Clustering algorithms guide
│   └── README.md               # Documentation index
├── data/                       # Data files (git-ignored)
│   ├── dataframes/             # Source dataframes
│   ├── entity_glossary/        # Organization name resolution
│   ├── graph_docs/             # Extracted relationship data
│   └── keys/                   # API configuration files
├── evaluation/                 # Evaluation files
├── results/                    # Output results
├── utils/                      # Utility modules
│   └── cluster/                # ⭐ Clustering algorithms
│       ├── cluster_algorithms.py  # Vector-based clustering
│       └── graph_clustering.py    # Graph-based clustering
├── .env                        # ⭐ Environment configuration (git-ignored)
├── .env.template               # Configuration template
├── app.py                      # 🚀 Streamlit web application
├── innovation_resolution.py    # 🚀 Main analysis script
├── evaluation.py               # Evaluation module
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

**💡 See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed structure documentation.**

## ⚙️ Setup

### Quick Setup (Recommended)

1. **Copy configuration template:**
   ```bash
   cp .env.template .env
   ```

2. **Edit `.env` with your API keys:**
   ```bash
   AZURE_OPENAI_API_KEY=your-key-here
   AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test configuration:**
   ```bash
   python tests/test_azure_connection.py
   ```

**📖 Detailed setup instructions: [docs/CONFIGURATION.md](docs/CONFIGURATION.md)**

### Legacy Configuration (Optional)

You can also use `data/keys/azure_config.json`:
```json
{
  "gpt-4o-mini": {
    "api_key": "YOUR_API_KEY",
    "api_base": "YOUR_AZURE_ENDPOINT",
    "api_version": "API_VERSION",
    "deployment": "gpt-4o-mini",
    "emb_deployment": "text-embedding-3-small"
  }
}
```

## 🚀 Usage

### Basic Usage

1. **Explore the data:**
   ```bash
   jupyter notebook introduction_data.ipynb
   ```

2. **Run innovation resolution:**
   ```bash
   python innovation_resolution.py
   ```

3. **Launch web interface:**
   ```bash
   streamlit run app.py
   ```

### Command Line Options

```bash
python innovation_resolution.py [options]

Options:
  --cache-type TYPE      Cache type (default: embedding)
  --cache-backend TYPE   Cache backend (json or memory, default: json)
  --cache-path PATH      Cache file path (default: ./embedding_vectors.json)
  --no-cache             Disable caching
  --skip-eval            Skip evaluation step
  --auto-label           Auto-label for evaluation
```

**Examples:**
```bash
# Standard run
python innovation_resolution.py

# Fast run (skip evaluation)
python innovation_resolution.py --skip-eval

# Custom cache location
python innovation_resolution.py --cache-path "./data/cache/embeddings.json"

# No caching (regenerate all)
python innovation_resolution.py --no-cache
```

## 🔧 What the Script Does

1. ✅ Load and combine data from multiple sources
2. ✅ Generate embeddings using Azure OpenAI
3. ✅ Detect duplicate innovations using semantic similarity
4. ✅ Create consolidated knowledge graph
5. ✅ Analyze innovation network
6. ✅ Generate visualizations
7. ✅ Export results to `results/` directory
8. ✅ Run evaluation metrics

## 🧪 Testing

```bash
# Test Azure API connection
python tests/test_azure_connection.py

# Test clustering algorithms
python tests/test_cluster.py
```

## 🏗️ Solution Architecture

* Embedding generation via `AzureOpenAIEmbeddings`
* Language understanding and chat responses via `AzureChatOpenAI`

#### Purpose of `AzureOpenAIEmbeddings`

The `AzureOpenAIEmbeddings` component is responsible for converting innovation-related textual descriptions into high-dimensional vector representations (dimension = 3072). These embeddings capture semantic meaning and are used to measure similarity between different innovations. The resulting vectors are stored in Azure AI Search for later retrieval and clustering. This forms the core of the semantic deduplication and search pipeline.

Deployment requirements:

* `text-embedding-3-large`&#x20;

#### Purpose of `AzureChatOpenAI`

The `AzureChatOpenAI` model is designed to answer user questions about innovation relationships. It retrieves the most relevant context from the vectorized innovation database stored in Azure AI Search and generates an intelligent, context-aware response. It uses LangChain's prompt templating and pipeline to ensure answers are grounded in retrieved information.

Deployment requirements:

* ⚠️ Resource type should be `Azure OpenAI` instead of `Azure Foundry` , in this way you get full access to OpenAI model

- `gpt-4.1-mini` or similar deployed under your Azure OpenAI resource. Change the name in **Configuration File** accordingly
- The inference task of model is `ChatCompletion`

### 2. Azure AI Search

Used for:

* Storing and querying innovation information embeddings
* Supporting similarity search and  information retrieve

Required setup:

* Create a search service instance
* Copy your API Key and Endpoint of Azure AI Search to **Configuration File**

---

## Coming Soon

* Support for Azure CosmosDB (for graph persistence)
* Automated Azure Search index provisioning
* Azure Functions / Logic Apps integration


### Innovation Resolution

The solution uses semantic similarity through embeddings to identify when different sources are discussing the same innovation:

1. For each innovation, we create a feature representation combining:
   - Innovation name
   - Innovation descriptions
   - Organizations that developed it

2. These features are converted to embeddings using OpenAI's embedding API or using TF-IDF as fallback

3. Cosine similarity is computed between all innovation pairs

4. Innovations with similarity above a threshold (default: 0.85) are considered duplicates

5. Duplicate innovations are mapped to a canonical innovation ID


### Innovation Duplicate Detection & Knowledge Graph

A lightweight pipeline to detect semantically duplicate innovations via embedding-based clustering and build a consolidated innovation-organization knowledge graph.

**Key Features:**
- Multiple clustering algorithms (HDBSCAN, K-Means, Agglomerative, Spectral)
- Graph-based clustering (Threshold, K-Core)
- Unified API interface via `cluster_with_stats()`
- Automatic noise detection and statistics

**Performance Benchmarks** (~2000 innovation samples):

| Method | Innovations | Organizations | Relationships | Notes |
|--------|------------|---------------|---------------|-------|
| Threshold | 1911 | 2490 | 12502 | Baseline |
| HDBSCAN | 1735 | 2490 | 12341 | Most conservative |
| K-Means | 1911 | 2490 | 12544 | Stable results |
| Agglomerative | 1911 | 2490 | 12544 | Hierarchical |
| Spectral | 1911 | 2490 | 12612 | Complex structures |

**See [docs/CLUSTERING.md](docs/CLUSTERING.md) for detailed algorithm documentation and usage examples.**


### Caching System

The solution uses a modular caching system for embeddings to improve performance:

1. **Architecture**:
   - Abstract `CacheBackend` protocol for different backend implementations
   - `JsonFileCache`: Persistent file-based caching (default)
   - `MemoryCache`: Fast in-memory caching
   - `EmbeddingCache`: Unified front-end with configurable backend
   - `CacheFactory`: Factory for creating cache instances

2. **Features**:
   - Automatic embedding caching to avoid redundant API calls
   - Configurable cache backend (file-based or in-memory)
   - Option to disable caching completely
   - Automatic recovery from cache loading errors

3. **Extensibility**:
   - Designed to be easily extended with new cache backends
   - Compatible with both OpenAI embeddings and TF-IDF fallback

### Knowledge Graph Consolidation

Once duplicates are identified, we consolidate the knowledge graph:

1. All information about duplicate innovations is merged into a single representation
2. The consolidated graph maintains:
   - Multiple names for the same innovation
   - Multiple descriptions
   - All organizations involved in development
   - All source documents mentioning the innovation
   - Original IDs of the duplicate innovations

### Network Analysis

The solution analyzes the consolidated innovation network to extract insights:

1. Basic statistics about innovations and organizations
2. Identification of innovations mentioned in multiple sources
3. Key organizations based on network centrality
4. Visualization of the innovation network

### Evaluation Module

The solution includes a comprehensive evaluation module that assesses the quality of the innovation resolution process through four main components:

1. **Consistency Checking**:
   - Randomly samples merged innovations for human verification
   - Generates a CSV file with innovation IDs, aliases, and source snippets
   - Allows human labelers to mark whether the merged items are truly the same innovation
   - Calculates the overall consistency rate (percentage of correctly merged innovations)
   - With `--auto-label`, automatically labels samples using LLM or heuristic methods

2. **Entity & Relation Extraction Accuracy**:
   - Compares automatically extracted entities and relations against human-annotated gold standards
   - Calculates precision, recall, and F1 score for both entity and relation extraction
   - Requires gold standard files (`gold_entities.json` and `gold_relations.json`) in the `evaluation` directory
   - With `--auto-label`, automatically generates gold standard files by sampling from predictions

3. **Knowledge Graph Structure Metrics**:
   - Calculates comprehensive graph statistics including:
     - Node and edge counts by type
     - Average degree and graph density
     - Connected component analysis
     - Connectivity ratio

4. **End-to-End QA Testing**:
   - Performs sample queries on the knowledge graph, such as:
     - "Which organizations developed innovation X?"
     - "What innovations are associated with organization Y?"
   - Saves example query results for manual inspection

### Template Files for Evaluation

The repository includes template files in the `evaluation` directory to help users get started with the evaluation process:

1. **`gold_entities_template.json`**: 
   - Example format for gold standard entity annotations
   - Copy this file to `evaluation/gold_entities.json` and expand with your own annotations

2. **`gold_relations_template.json`**: 
   - Example format for gold standard relation annotations
   - Copy this file to `evaluation/gold_relations.json` and expand with your own annotations

3. **`consistency_sample_template.csv`**: 
   - Example of the consistency checking CSV with sample labels
   - Shows how to fill in the `human_label` column with "Yes" or "No"

4. **`qa_examples_template.json`**: 
   - Example of the QA testing results format
   - Shows the expected structure of innovation-organization relationships

To use these templates:

```bash
# For entity evaluation
cp evaluation/gold_entities_template.json evaluation/gold_entities.json
# Edit evaluation/gold_entities.json with your annotations

# For relation evaluation
cp evaluation/gold_relations_template.json evaluation/gold_relations.json
# Edit evaluation/gold_relations.json with your annotations

# For consistency checking (after first run generates the sample)
# Edit evaluation/consistency_sample.csv adding Yes/No in the human_label column
# Then run the script again to calculate consistency rate
```

To use the evaluation module, you can:

1. **Prepare Gold Standards** (optional):
   - Create `evaluation/gold_entities.json` with the format: `[{"name": "...", "type": "Innovation"}, ...]`
   - Create `evaluation/gold_relations.json` with the format: `[{"innovation": "...", "organization": "...", "relation": "DEVELOPED_BY"}, ...]`

2. **Run the Evaluation**:
   - Execute `python innovation_resolution.py` to run the complete pipeline with evaluation
   - For the first run, consistency checking samples will be generated
   - Fill in the `human_label` column in the generated CSV file
   - Run the script again to calculate the consistency rate

3. **Review Results**:
   - Evaluation metrics are printed to console during execution
   - Complete evaluation results are saved to `evaluation/evaluation_results.json`
   - QA examples are saved to `evaluation/qa_examples.json`

## 📊 Results

The solution produces outputs in the `results/` directory:

### Data Files
- `canonical_mapping.json` - Innovation ID mappings
- `consolidated_graph.json` - Complete knowledge graph
- `innovation_stats.json` - Network statistics
- `multi_source_innovations.json` - Cross-source innovations
- `key_nodes.json` - Key organizations and innovations

### Visualizations
- `innovation_network.png` - Network visualization
- `innovation_network_3d.html` - Interactive 3D network
- `innovation_stats.png` - Statistics charts
- `top_organizations.png` - Top organizations

### Evaluation (in `evaluation/` directory)
- `consistency_sample.csv` - Manual checking samples
- `evaluation_results.json` - Evaluation metrics
- `qa_examples.json` - QA examples
- `pred_entities.json` - Predicted entities
- `pred_relations.json` - Predicted relations

## 🔗 Additional Resources

- **[Full Documentation](docs/README.md)** - Complete documentation index
- **[YouTube Demo](https://www.youtube.com/watch?v=yKNr22bu9Yc)** - Video walkthrough
- **[Live App](https://innovation-duplication.streamlit.app)** - Try it online

## 🛠️ Technology Stack

- **Data Processing:** pandas, numpy, pydantic
- **AI/ML:** Azure OpenAI, LangChain, scikit-learn
- **Visualization:** matplotlib, seaborn, plotly, networkx
- **Web Interface:** Streamlit
- **Clustering:** HDBSCAN

See [`requirements.txt`](requirements.txt) for complete dependencies.

## 👥 Contributors

This project is part of the AaltoAI Hackathon in collaboration with VTT.

## 📄 License

[Add license information]

---

**📚 [Documentation](docs/README.md)** | **🚀 [Getting Started](docs/GETTING_STARTED.md)** | **🛠️ [Development Guide](docs/DEVELOPMENT.md)**

