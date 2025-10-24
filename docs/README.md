# Documentation Index

Welcome to the documentation for the **Innovation Platform**. The guides below have been consolidated so you can find the essentials quickly.

## 🎯 Start Here
- **[GETTING_STARTED.md](GETTING_STARTED.md)** — Install dependencies, configure Azure keys, run your first pipeline.
- **[DEVELOPMENT.md](DEVELOPMENT.md)** — Project architecture, data pipeline modules, evaluation and testing tips.

## ⚙️ Operate the CLI & Automations
- **[CLI_USAGE_GUIDE.md](CLI_USAGE_GUIDE.md)** — One-stop reference for pipeline flags, scripted workflows, and automation patterns.

## 📊 Understand Outputs
- **[ANALYSIS_RESULTS_CHEATSHEET.md](ANALYSIS_RESULTS_CHEATSHEET.md)** — What each artifact contains, how to export bespoke reports, and how to serve the static dashboard.

## 🧠 Methodology
- **[CLUSTERING_GUIDE.md](CLUSTERING_GUIDE.md)** — Clustering strategies (HDBSCAN, K-Means, graph methods) and when to use them.
- **[DATA_STRUCTURES.md](DATA_STRUCTURES.md)** — Core node/relationship schemas powering the knowledge graph.

## 📚 Additional References
- **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)** — Algorithmic deep dive and design rationale.
- **[DATA_PIPELINE_REFACTORING_GUIDE.md](DATA_PIPELINE_REFACTORING_GUIDE.md)** — Historical refactor notes; kept for context but key points are summarized in `DEVELOPMENT.md`.

> **Tip:** Need a quick command? Jump to the task-based checklist in `CLI_USAGE_GUIDE.md`. Looking for specific output files? See the tables in `ANALYSIS_RESULTS_CHEATSHEET.md`.

## 🎯 Key Sections

### Clustering Guide

**Target Audience:** Users wanting to understand or tune clustering  ### Getting Started Guide

**Content:**- ✅ Installation & setup

- Supported clustering methods (HDBSCAN, K-means, etc.)- ✅ Configuration details

- When to use each method- ✅ First run guide

- Performance comparison- ✅ Troubleshooting

- Configuration examples- ✅ Migration from JSON



### Technical Details### Development Guide  

**Target Audience:** Developers, technical users  - ✅ Project structure

**Content:**- ✅ Core components

- Core algorithm: `resolve_innovation_duplicates`- ✅ Data flow

- Data flow and processing steps- ✅ API usage

- Embedding generation and caching- ✅ Development workflow

- Information preservation mechanisms- ✅ Code style & security



### Development Guide## 🆘 Need Help?

**Target Audience:** Contributors, maintainers  

**Content:****Configuration issues?**  

- Project structure→ See [GETTING_STARTED.md - Troubleshooting](GETTING_STARTED.md#troubleshooting)

- Code organization

- Development workflow**Understanding code structure?**  

- Testing approach→ See [DEVELOPMENT.md - Project Structure](DEVELOPMENT.md#project-structure)



### Data Pipeline Refactoring Guide**API usage questions?**  

**Target Audience:** Architects, advanced developers  → See [DEVELOPMENT.md - API Usage](DEVELOPMENT.md#api-usage)

**Content:**

- Modular architecture design**Something else?**  

- Strategy pattern implementation→ Check the [main README](../README.md)

- Cache system design

- Best practices---



## 🎯 Learning Paths**Back to [Main README](../README.md)**



### Path 1: Quick Start (30 minutes)
1. [Getting Started](GETTING_STARTED.md) - Setup and run
2. [Clustering Guide](CLUSTERING_GUIDE.md) - Understand the core algorithm

### Path 2: Developer Onboarding (1 hour)
1. [Getting Started](GETTING_STARTED.md) - Environment setup
2. [Development Guide](DEVELOPMENT.md) - Project structure
3. [Technical Details](TECHNICAL_DETAILS.md) - Core implementation
4. [Data Pipeline Refactoring Guide](DATA_PIPELINE_REFACTORING_GUIDE.md) - Architecture

### Path 3: Advanced Deep Dive (2 hours)
1. All documents above
2. Source code review in `/data/processors/`
3. Test code in `/tests/`

## 🔧 Key Concepts

### Innovation Deduplication
The system identifies and merges duplicate innovations mentioned across different data sources using semantic similarity analysis.

### Clustering Methods
- **HDBSCAN** (default): Density-based clustering, auto-detects number of clusters
- **K-means**: Requires pre-specified number of clusters
- **Graph-based**: Uses similarity graphs (advanced usage)

### Data Processing Pipeline
1. Load data from multiple sources (CSV + GraphDocuments)
2. Extract unique innovations
3. Build feature representations
4. Generate embeddings (with caching)
5. Cluster similar innovations
6. Consolidate knowledge graph

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Cunyli/Innovation-Duplication/issues)
- **Discussions**: Use GitHub Discussions for questions

## 🔄 Document Status

| Document | Last Updated | Status |
|----------|-------------|---------|
| Getting Started | 2025-10 | ✅ Current |
| Clustering Guide | 2025-10 | ✅ Current |
| Technical Details | 2025-10 | ✅ Current |
| Development Guide | 2025-10 | ✅ Current |
| Data Pipeline Refactoring | 2025-10 | ✅ Current |
