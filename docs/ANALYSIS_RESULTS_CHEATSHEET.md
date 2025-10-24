# Outputs & Export Cheat Sheet

This guide summarises the artefacts produced by the pipeline and how to extend or re-export them programmatically.

## 📁 What ends up in `results/`

| File | Description | When it appears |
|------|-------------|-----------------|
| `canonical_mapping.json` | Map from raw innovation IDs → canonical IDs | After clustering |
| `consolidated_graph.json` | Merged innovations, organisations, relationships | After graph build |
| `innovation_stats*.{json,png}` | Aggregate metrics for dashboards | During `analyze` / `visualize` |
| `innovation_network_tufte_*.{png,html}` | 2D/3D network snapshots | During `visualize` |
| `top_organizations*.png` | Horizontal bar chart of prolific organisations | During `visualize` |
| `multi_source_innovations.json` | Innovations supported by multiple sources | During `export` |

`--output-dir` changes the destination; everything keeps the same filenames inside the new folder.

## 🔍 `analysis_results` structure (returned by `analyze_innovation_network`)

```python
{
    "graph": nx.Graph,                  # innovation + organisation nodes
    "stats": {...},                     # totals, averages, ratios
    "multi_source": {innovation_id: {...}},
    "top_orgs": [(org_id, count), ...],
    "key_orgs": [(org_id, centrality), ...],
    "key_innovations": [(inno_id, centrality), ...]
}
```

NetworkX node attributes:
- Innovations: `{"type": "Innovation", "names": set[str], "sources": set[str], ...}`
- Organisations: `{"type": "Organization", "name": str, ...}`

Edges carry `type = "DEVELOPED_BY"` or `"COLLABORATION"`.

## 🧾 Using the Result Exporter programmatically

```python
from innovation_platform.data_pipeline.processors import ResultExporter
from pathlib import Path

exporter = ResultExporter(output_dir=Path("./results/custom"))
exporter.save_analysis(analysis_results)
exporter.save_graph(consolidated_graph)
exporter.save_canonical_mapping(canonical_mapping)
```

Helper function:
```python
from innovation_platform.data_pipeline.processors import export_analysis_results

export_analysis_results(analysis_results, consolidated_graph, canonical_mapping,
                        output_dir="./results/custom")
```

The helpers automatically convert `set` values to lists so JSON serialisation works out of the box.

## 🌐 Serving static outputs

```bash
./start_server.sh 8000   # pass another port if 8000 is busy
```

- Validates that `results/` (or the selected `--output-dir`) exists.
- Launches `python -m http.server` and prints direct links such as:
  - `http://localhost:8000/innovation_network_tufte_3D.html`
  - `http://localhost:8000/top_organizations_tufte.png`

## 🧮 Common analysis snippets

```python
from pathlib import Path
import json

results_dir = Path("results")
with open(results_dir / "canonical_mapping.json", "r", encoding="utf-8") as f:
    canonical = json.load(f)

print(f"Unique canonical innovations: {len(set(canonical.values()))}")
```

```python
import networkx as nx
from innovation_platform.innovation_resolution import analyze_innovation_network

analysis_results = analyze_innovation_network(consolidated_graph,
                                              top_n=20, max_iter=1000,
                                              print_summary=True)
G = analysis_results["graph"]
print(nx.number_connected_components(G))
```

## 🧪 Evaluation artefacts

When `--skip-eval` is **not** used, expect these under `evaluation/` (or `--eval-dir`):

| File | Purpose |
|------|---------|
| `pred_entities.json` / `pred_relations.json` | Pipeline predictions |
| `gold_entities_template.json` / `gold_relations_template.json` | Fill manually for precision/recall |
| `consistency_sample.csv` | Randomly sampled merged innovations for manual QA |
| `evaluation_results.json` | Aggregate scores from `run_all_evaluations` |

Setting `--auto-label` enables LLM-assisted labelling (Azure consumption applies).

## 🧰 Troubleshooting

- Missing PNG/HTML assets → rerun with `--steps analyze visualize export`.
- Empty `canonical_mapping.json` → ensure embeddings exist (check cache path) and clustering produced non-empty results.
- `start_server.sh` complains about missing directory → pipeline hasn’t generated outputs yet; run the CLI first.

For more details on command-line flags see `CLI_USAGE_GUIDE.md`. For visual styling hooks check `src/innovation_platform/vis.py`.
