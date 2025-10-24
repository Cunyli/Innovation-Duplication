# CLI & Automation Guide

The pipeline exposes a single entrypoint: `innovation_platform.innovation_resolution`. This guide merges the former quick reference, usage guide, and refactoring notes into one place.

## 🚀 Task-based Quick Reference

| Goal | Command | Notes |
|------|---------|-------|
| Full run (default workflow) | `PYTHONPATH=src python -m innovation_platform.innovation_resolution` | Executes load → dedupe → graph → analyse → visualise → export → evaluate |
| Fast iteration (skip visuals & eval) | `PYTHONPATH=src python -m innovation_platform.innovation_resolution --skip-visualization --skip-export --skip-eval` | Use when tuning clusters |
| Graph-based clustering | `PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method graph_threshold --similarity-threshold 0.88` | Threshold defaults to cosine similarity |
| Regenerate only visuals & exports | `PYTHONPATH=src python -m innovation_platform.innovation_resolution --steps analyze visualize export --skip-eval --no-cache` | Reuses existing canonical mapping |
| Evaluation only | `PYTHONPATH=src python -m innovation_platform.innovation_resolution --steps eval --skip-visualization --skip-export` | Assumes `results/` contains consolidated data |
| Custom output location | `PYTHONPATH=src python -m innovation_platform.innovation_resolution --output-dir ./runs/2025-01-31` | Directory is created if missing |

Run `--help` to see every flag grouped by category.

## ⚙️ Core Arguments

### Data locations
```
--data-dir DATA_DIR
--graph-docs-company PATH
--graph-docs-vtt PATH
```
Defaults point at the checked-in `data/` subtree. Override when running on alternate datasets.

### Clustering controls
```
--clustering-method {hdbscan,kmeans,agglomerative,spectral,graph_threshold,graph_kcore}
--min-cluster-size N
--metric {cosine,euclidean,manhattan}
--n-clusters N              # for k-means / agglomerative / spectral
--similarity-threshold X    # for graph_threshold
--k-core N                  # for graph_kcore
```
HDBSCAN is the default. Graph methods require a similarity threshold or k-core depth.

### Workflow toggles
```
--steps load cluster graph analyze visualize export eval
--skip-visualization
--skip-export
--skip-eval
--quiet / --verbose
```
`--steps` lets you run a subset in order; combine with skip flags for finer control.

### Cache & embeddings
```
--cache-type embedding
--cache-backend {json,memory}
--cache-path ./embedding_vectors.json
--no-cache
```
The JSON backend keeps embeddings between runs. Use `--no-cache` when testing alternative feature builders.

### Output & evaluation
```
--output-dir ./results
--eval-dir ./evaluation
--auto-label               # allow LLM-assisted evaluation (requires Azure credits)
```
`--output-dir` propagates to visualisations and exported JSON summaries.

## 🧭 Recommended Workflows

### 1. Start fresh with cached embeddings
```bash
PYTHONPATH=src python -m innovation_platform.innovation_resolution --cache-path ./embedding_vectors.json
```
- First run populates the cache; subsequent runs reuse it.
- Delete or change `--cache-path` when switching datasets.

### 2. Tune clustering hyperparameters
```bash
PYTHONPATH=src python -m innovation_platform.innovation_resolution \
  --clustering-method hdbscan \
  --min-cluster-size 4 \
  --metric cosine \
  --skip-visualization --skip-export --skip-eval
```
Inspect the log summary (`簇数量`, `噪音点`) and adjust parameters accordingly.

### 3. Generate artefacts for presentation
```bash
PYTHONPATH=src python -m innovation_platform.innovation_resolution \
  --steps analyze visualize export \
  --output-dir ./results/pitch_deck \
  --skip-eval
./start_server.sh
```
Serve the HTML/PNG files from the new output directory using the helper script (pass a port if 8000 is busy).

### 4. Batch experiments
Create a shell script to sweep parameters:
```bash
for threshold in 0.85 0.88 0.90; do
  PYTHONPATH=src python -m innovation_platform.innovation_resolution \
    --clustering-method graph_threshold \
    --similarity-threshold ${threshold} \
    --output-dir ./runs/graph-${threshold} \
    --skip-eval --skip-visualization
done
```
Follow up with `--steps analyze visualize` on the most promising run.

## 🗂️ Integration Touchpoints

- **Streamlit app**: The app imports `chat_bot` from `innovation_platform.innovation_resolution`. Run the CLI first so the app can find generated artefacts in `results/`.
- **Scripts**: Utilities in `scripts/` (e.g., `analyze_clustering_usage.py`) honour the new `src` layout by appending `PYTHONPATH=src` when executed.
- **Automation**: When scheduling via cron or GitHub Actions, remember to export `PYTHONPATH=src` and point `AZURE_*` env vars at secure secrets storage.

## ✅ Checklist Before Running in Production

- [ ] `.env` has the correct Azure deployment names and endpoint.
- [ ] `data/` is mounted or synced to the runtime environment.
- [ ] Embedding cache location is writable (or `--no-cache` is intentional).
- [ ] Desired output directory is empty or you are comfortable overwriting.
- [ ] Optional services (Azure AI Search) are configured if the Streamlit chatbot is part of the deliverable.

Need a deeper explanation of what each flag impacts? Pair this guide with `ANALYSIS_RESULTS_CHEATSHEET.md` (outputs) and `DEVELOPMENT.md` (architecture).
