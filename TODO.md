# TODO / Future Enhancements

### Query & Experience
- Wrap the semantic query engine in REST or GraphQL endpoints so external systems can retrieve innovations and organisations programmatically.
- Expose the Streamlit panels as reusable modules (chat, network explorer, metrics) and document the configuration needed to run them on managed hosting (Streamlit Cloud / Azure App Service).

### Evaluation & Quality
- Design an evaluation harness that scores clustering and deduplication quality (purity, silhouette, pairwise precision/recall) using labelled subsets or synthetic benchmarks.
- Extend the evaluation toolkit to benchmark query relevance: create a suite of example questions and expected answers to validate the chatbot/search layer after each release.
- Build regression tests that re-run a small anonymised dataset to verify canonical mappings, cluster assignments, and evaluation metrics remain stable.
- Stand up a full validation pipeline: sampling + manual labelling for dedupe consistency, gold JSONs for entity/relationship extraction (with precision/recall/F1), automatic graph-structure stats, and a QA sample report—then expose it via `pipeline_runner --with-eval` so one flag runs the whole evaluation chain.
- Develop a richer vectorisation strategy: blend dense embeddings with sparse (TF-IDF/BM25) features, encode structured metadata (domain/time/source), and add centroid-based sanity checks to drop outlier nodes after clustering.
- Evaluate alternative / domain-adapted embedding models (e.g. Jina, Cohere, Voyage, lightweight LoRA fine-tunes) and track embedding versions so the pipeline can auto-rebuild when the model changes.
- Implement hard-negative mining / lightweight Siamese fine-tuning to calibrate similarity scores, plus post-cluster validation (recomputing centroids, re-checking thresholds) to reduce over-merging.

### Pipeline & Infrastructure
- Add a command that batches multiple clustering strategies in one run and outputs a comparison dashboard (precision/recall vs. runtime) for faster hyperparameter tuning.
- Extend `ResultExporter` to write to cloud storage (Azure Blob, S3) and optionally publish artefacts as versioned datasets.
- Introduce data QA checks before the run (schema validation, column presence, basic stats) to catch issues early.
- Explore asynchronous embedding generation so the main pipeline can queue new documents while reusing cached vectors.
- Package the system in Docker (with optional GPU support) to simplify deployment in cloud environments.
<<<<<<< Updated upstream
=======
- Add interactive highlighting for the 3D network viz (hover/click to brighten a node + its edges, dim the rest) without sacrificing camera responsiveness; keep this behaviour in sync between standalone HTML files and the Streamlit embedding.
- Provide a Dockerised graph database (e.g. Neo4j) setup plus an exporter so each pipeline run can push the consolidated graph into the DB for downstream querying/visualisation.
>>>>>>> Stashed changes

### Advanced Methods
- Evaluate modern representation learning techniques (e.g., graph neural networks, transformer-based entity matching) as drop-in replacements or augmentations for the current embedding + clustering approach.
- Experiment with hybrid approaches for deduplication that combine rule-based signals with learnable similarity models (contrastive learning, metric learning) to boost robustness.
