# TODO / Future Enhancements

### Query & Experience
- Wrap the semantic query engine in REST or GraphQL endpoints so external systems can retrieve innovations and organisations programmatically.
- Expose the Streamlit panels as reusable modules (chat, network explorer, metrics) and document the configuration needed to run them on managed hosting (Streamlit Cloud / Azure App Service).

### Evaluation & Quality
- Design an evaluation harness that scores clustering and deduplication quality (purity, silhouette, pairwise precision/recall) using labelled subsets or synthetic benchmarks.
- Extend the evaluation toolkit to benchmark query relevance: create a suite of example questions and expected answers to validate the chatbot/search layer after each release.
- Build regression tests that re-run a small anonymised dataset to verify canonical mappings, cluster assignments, and evaluation metrics remain stable.

### Pipeline & Infrastructure
- Add a command that batches multiple clustering strategies in one run and outputs a comparison dashboard (precision/recall vs. runtime) for faster hyperparameter tuning.
- Extend `ResultExporter` to write to cloud storage (Azure Blob, S3) and optionally publish artefacts as versioned datasets.
- Introduce data QA checks before the run (schema validation, column presence, basic stats) to catch issues early.
- Explore asynchronous embedding generation so the main pipeline can queue new documents while reusing cached vectors.
- Package the system in Docker (with optional GPU support) to simplify deployment in cloud environments.

### Advanced Methods
- Evaluate modern representation learning techniques (e.g., graph neural networks, transformer-based entity matching) as drop-in replacements or augmentations for the current embedding + clustering approach.
- Experiment with hybrid approaches for deduplication that combine rule-based signals with learnable similarity models (contrastive learning, metric learning) to boost robustness.
