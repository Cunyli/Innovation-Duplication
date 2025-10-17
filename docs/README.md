# Documentation Index# Documentation



Welcome to the VTT Innovation Resolution documentation!Complete documentation for the Innovation-Duplication project.



## 📖 Core Documentation## 📚 Documentation Files



### For Users| Document | Description | Read Time |

|----------|-------------|-----------|

| Document | Description | Read Time || **[GETTING_STARTED.md](GETTING_STARTED.md)** | Setup, configuration, and first run | 10 min |

|----------|-------------|-----------|| **[DEVELOPMENT.md](DEVELOPMENT.md)** | Project structure and development guide | 15 min |

| **[Getting Started](GETTING_STARTED.md)** | Setup, installation, and first run | 10 min || **[DATA_PIPELINE_REFACTORING_GUIDE.md](DATA_PIPELINE_REFACTORING_GUIDE.md)** | Data pipeline modularization guide | 20 min |

| **[Clustering Guide](CLUSTERING_GUIDE.md)** | Understanding clustering methods | 10 min || **[CLUSTERING.md](CLUSTERING.md)** | Graph clustering algorithms | 10 min |



### For Developers## 🚀 Quick Links



| Document | Description | Read Time |### I want to...

|----------|-------------|-----------|- **Get started quickly** → [GETTING_STARTED.md](GETTING_STARTED.md)

| **[Development Guide](DEVELOPMENT.md)** | Project structure & workflow | 15 min |- **Understand the code** → [DEVELOPMENT.md](DEVELOPMENT.md#project-structure)

| **[Technical Details](TECHNICAL_DETAILS.md)** | Deep dive into core algorithms | 20 min |- **Learn about refactoring** → [DATA_PIPELINE_REFACTORING_GUIDE.md](DATA_PIPELINE_REFACTORING_GUIDE.md)

| **[Data Pipeline Refactoring Guide](DATA_PIPELINE_REFACTORING_GUIDE.md)** | Architecture & design patterns | 20 min |- **Configure API keys** → [GETTING_STARTED.md](GETTING_STARTED.md#configure-environment)

- **Test my setup** → [GETTING_STARTED.md](GETTING_STARTED.md#test-configuration)

## 🚀 Quick Navigation- **Troubleshoot issues** → [GETTING_STARTED.md](GETTING_STARTED.md#troubleshooting)

- **Add new features** → [DEVELOPMENT.md](DEVELOPMENT.md#adding-new-features)

**I want to...**

## 📖 Reading Guide

- **Get started quickly** → [Getting Started](GETTING_STARTED.md)

- **Understand clustering** → [Clustering Guide](CLUSTERING_GUIDE.md)### New Users (Start Here!)

- **Understand the code** → [Technical Details](TECHNICAL_DETAILS.md)1. Read [GETTING_STARTED.md](GETTING_STARTED.md) (10 min)

- **Contribute code** → [Development Guide](DEVELOPMENT.md)2. Follow the setup steps

- **Learn the architecture** → [Data Pipeline Refactoring Guide](DATA_PIPELINE_REFACTORING_GUIDE.md)3. Run your first analysis

4. Explore the results

## 📚 Document Overview

### Developers

### Getting Started Guide1. Review [GETTING_STARTED.md](GETTING_STARTED.md) for setup

**Target Audience:** New users, data scientists  2. Study [DEVELOPMENT.md](DEVELOPMENT.md) for architecture

**Content:**3. Read [DATA_PIPELINE_REFACTORING_GUIDE.md](DATA_PIPELINE_REFACTORING_GUIDE.md) for data pipeline modules

- Environment setup4. Check component-specific READMEs:

- Configuration   - [config/README.md](../config/README.md) - Configuration API

- Running the pipeline   - [tests/README.md](../tests/README.md) - Test suite

- Troubleshooting

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
