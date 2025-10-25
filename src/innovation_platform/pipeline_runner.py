#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline runner that orchestrates the entire workflow:
data loading → deduplication → graph build → analysis → visualisation →
export → query engine warm-up → (optional) evaluation.

It can resume from existing artefacts (consolidated graph + canonical mapping)
to avoid rerunning heavy steps.

To run this file:
PYTHONPATH=src python -m innovation_platform.pipeline_runner --resume --sample-query "battery recycling"

with --force will ignore snapshot, force to run against the entire pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .data_pipeline.processors import initialize_openai_client
from .evaluation import run_all_evaluations
from .innovation_resolution import (
    analyze_innovation_network,
    create_innovation_knowledge_graph,
    export_results,
    load_and_combine_data,
    resolve_innovation_duplicates,
    visualize_network_tufte,
)
from .local_entity_processing import Node, Relationship
from .innovation_resolution import create_query_engine

# Step order is important
DEFAULT_STEPS = ["load", "dedupe", "graph", "analyze", "visualize", "export", "query", "eval"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "evaluation"
STATE_PATH = PROJECT_ROOT / ".pipeline_state.json"
INPUT_SNAPSHOT_PATH = PROJECT_ROOT / ".pipeline_inputs.json"

TRACKED_INPUTS = [
    PROJECT_ROOT / "data" / "dataframes",
    PROJECT_ROOT / "data" / "graph_docs_names_resolved",
    PROJECT_ROOT / "data" / "graph_docs_vtt_domain_names_resolved",
]


@dataclass
class PipelineContext:
    df_relationships: Optional["pd.DataFrame"] = None  # type: ignore[name-defined]
    all_pred_entities: List[Dict] = field(default_factory=list)
    all_pred_relations: List[Dict] = field(default_factory=list)
    canonical_mapping: Optional[Dict[str, str]] = None
    consolidated_graph: Optional[Dict] = None
    analysis_results: Optional[Dict] = None


class PipelineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir).resolve()
        self.eval_dir = Path(args.eval_dir).resolve()
        self.consolidated_graph_path = self.output_dir / "consolidated_graph.json"
        self.canonical_mapping_path = self.output_dir / "canonical_mapping.json"
        self.context = PipelineContext()
        self.llm = None
        self.embedding_model = None
        self.completed_steps: List[str] = []
        self.current_snapshot = self._collect_input_snapshot()
        self.previous_snapshot = self._load_previous_snapshot()
        self.input_changes = self._diff_snapshots(self.previous_snapshot, self.current_snapshot)
        self.snapshot_changed = bool(self.input_changes) or self.args.force

    def run(self) -> None:
        self._prepare_environment()
        skip_steps = self._maybe_resume()

        selected_steps = self.args.steps or DEFAULT_STEPS
        for step in DEFAULT_STEPS:
            if step not in selected_steps:
                continue
            if step in skip_steps:
                print(f"[skip] Step '{step}' already satisfied (resume).")
                self.completed_steps.append(step)
                continue
            handler = getattr(self, f"_step_{step}", None)
            if handler is None:
                raise ValueError(f"Unknown step '{step}'")
            handler()
            self.completed_steps.append(step)

        self._write_state()
        print("\n✅ Pipeline completed.")

    def _prepare_environment(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        if self.args.skip_openai:
            print("⚠️  Skipping OpenAI initialisation (TF-IDF fallback will be used).")
            return

        try:
            print("🔌 Initialising Azure OpenAI clients...")
            self.llm, self.embedding_model = initialize_openai_client()
            print("✅ OpenAI clients ready.")
        except Exception as exc:  # pragma: no cover - depends on env
            print(f"⚠️  Failed to initialise OpenAI clients: {exc}")
            print("    Falling back to TF-IDF embeddings.")
            self.llm = None
            self.embedding_model = None

    def _maybe_resume(self) -> Set[str]:
        if not (self.args.resume and not self.args.force):
            return set()

        if not (self.consolidated_graph_path.exists() and self.canonical_mapping_path.exists()):
            print("⚠️  Resume requested but required artefacts are missing. Running full pipeline.")
            return set()

        print("♻️  Resuming from existing consolidated graph and canonical mapping.")
        with self.consolidated_graph_path.open("r", encoding="utf-8") as f:
            self.context.consolidated_graph = json.load(f)
        with self.canonical_mapping_path.open("r", encoding="utf-8") as f:
            self.context.canonical_mapping = json.load(f)

        if not self.snapshot_changed:
            print("🛌  Input data unchanged. Skipping load/dedupe/graph/analyze/visualize/export.")
            return {"load", "dedupe", "graph", "analyze", "visualize", "export"}

        if self.input_changes:
            print(f"🔄 Detected {len(self.input_changes)} modified or new input files.")
        return {"load", "dedupe", "graph"}

    # ---- individual steps -------------------------------------------------

    def _step_load(self) -> None:
        print("\n[1/8] Loading and combining raw data...")
        df_relationships, all_pred_entities, all_pred_relations = load_and_combine_data()
        self.context.df_relationships = df_relationships
        self.context.all_pred_entities = all_pred_entities
        self.context.all_pred_relations = all_pred_relations

    def _step_dedupe(self) -> None:
        print("\n[2/8] Resolving innovation duplicates...")
        if self.context.df_relationships is None:
            raise RuntimeError("Relationships dataframe missing. Did you skip the load step?")

        cache_config = {
            "type": "embedding",
            "backend": "json",
            "path": str(Path(self.args.cache_path).resolve()),
            "use_cache": not self.args.force,
        }
        canonical_mapping = resolve_innovation_duplicates(
            self.context.df_relationships,
            model=self.embedding_model,
            cache_config=cache_config,
            method=self.args.clustering_method,
            **self.args.clustering_kwargs,
        )
        self.context.canonical_mapping = canonical_mapping
        self._save_predictions()

    def _step_graph(self) -> None:
        print("\n[3/8] Building consolidated knowledge graph...")
        if self.context.df_relationships is None or self.context.canonical_mapping is None:
            raise RuntimeError("Cannot build graph without relationships and canonical mapping.")
        self.context.consolidated_graph = create_innovation_knowledge_graph(
            self.context.df_relationships, self.context.canonical_mapping
        )

    def _step_analyze(self) -> None:
        print("\n[4/8] Analysing innovation network...")
        if self.context.consolidated_graph is None:
            raise RuntimeError("Consolidated graph missing.")
        self.context.analysis_results = analyze_innovation_network(
            self.context.consolidated_graph,
            top_n=self.args.top_n,
            max_iter=self.args.max_iter,
            print_summary=True,
        )

    def _step_visualize(self) -> None:
        if self.args.skip_visualization:
            print("[skip] Visualisation disabled via flag.")
            return
        print("\n[5/8] Generating visualisations...")
        if not self.context.analysis_results:
            raise RuntimeError("Analysis results missing.")
        visualize_network_tufte(self.context.analysis_results)

    def _step_export(self) -> None:
        if self.args.skip_export:
            print("[skip] Export disabled via flag.")
            return
        print("\n[6/8] Exporting artefacts...")
        if not (self.context.analysis_results and self.context.consolidated_graph and self.context.canonical_mapping):
            raise RuntimeError("Missing artefacts required for export.")
        export_results(
            self.context.analysis_results,
            self.context.consolidated_graph,
            self.context.canonical_mapping,
            output_dir=str(self.output_dir),
        )

    def _step_query(self) -> None:
        print("\n[7/8] Preparing semantic query engine...")
        if self.context.consolidated_graph is None:
            raise RuntimeError("Consolidated graph missing.")

        cache_config = {
            "type": "embedding",
            "backend": "json",
            "path": str(Path(self.args.query_cache_path).resolve()),
            "use_cache": not self.args.force,
        }
        engine = create_query_engine(
            consolidated_graph=self.context.consolidated_graph,
            embedding_model=self.embedding_model,
            cache_config=cache_config,
        )

        sample_query = self.args.sample_query.strip()
        if not sample_query:
            sample_query = "artificial intelligence innovation"

        results = engine.search(sample_query, top_k=self.args.top_k)
        print(f"Sample query: {sample_query}")
        if not results:
            print("No results found (try adjusting the query).")
            return
        for hit in results:
            print(
                f"- {hit.innovation_id:<25} score={hit.score:.3f} "
                f"orgs={', '.join(hit.organizations[:3])}"
            )

    def _step_eval(self) -> None:
        if not self.args.with_eval:
            print("[skip] Evaluation disabled. Use --with-eval to enable.")
            return
        print("\n[8/8] Running evaluation suite...")
        if self.context.consolidated_graph is None:
            raise RuntimeError("Consolidated graph missing.")

        merged_innovations = []
        for inno_id, inno_data in self.context.consolidated_graph["innovations"].items():
            node = Node(
                id=inno_id,
                type="Innovation",
                properties={
                    "aliases": "|".join(list(inno_data.get("names", []))),
                    "source_docs": "|".join(list(inno_data.get("descriptions", []))),
                    "developed_by": "|".join(list(inno_data.get("developed_by", []))),
                    "sources": "|".join(list(inno_data.get("sources", []))),
                },
            )
            merged_innovations.append(node)

        all_nodes = []
        for inno_id, inno_data in self.context.consolidated_graph["innovations"].items():
            all_nodes.append(
                Node(
                    id=inno_id,
                    type="Innovation",
                    properties={
                        "name": next(iter(inno_data.get("names", [inno_id])), inno_id),
                        "description": next(iter(inno_data.get("descriptions", [""])), ""),
                    },
                )
            )
        for org_id, org_data in self.context.consolidated_graph["organizations"].items():
            all_nodes.append(
                Node(
                    id=org_id,
                    type="Organization",
                    properties={
                        "name": str(org_data.get("name") or ""),
                        "description": str(org_data.get("description") or ""),
                    },
                )
            )

        all_rels = []
        for rel in self.context.consolidated_graph["relationships"]:
            all_rels.append(
                Relationship(
                    source=rel["source"],
                    source_type="Innovation"
                    if rel["source"] in self.context.consolidated_graph["innovations"]
                    else "Organization",
                    target=rel["target"],
                    target_type="Organization"
                    if rel["target"] in self.context.consolidated_graph["organizations"]
                    else "Innovation",
                    type=rel["type"],
                    properties={},
                )
            )

        run_all_evaluations(
            merged_innovations=merged_innovations,
            all_nodes=all_nodes,
            all_rels=all_rels,
            data_dir=str(PROJECT_ROOT / "data"),
            results_dir=str(self.output_dir),
            eval_dir=str(self.eval_dir),
            auto_label=self.args.auto_label,
            llm=self.llm,
        )

    # ---- helpers ----------------------------------------------------------

    def _save_predictions(self) -> None:
        if not (self.context.all_pred_entities and self.context.all_pred_relations):
            return

        eval_dir = self.eval_dir
        eval_dir.mkdir(parents=True, exist_ok=True)

        unique_entities = []
        seen_entities = set()
        for entity in self.context.all_pred_entities:
            key = (entity.get("name", "").lower(), entity.get("type"))
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(entity)

        unique_relations = []
        seen_relations = set()
        for relation in self.context.all_pred_relations:
            key = (
                relation.get("innovation", "").lower(),
                relation.get("organization", "").lower(),
                relation.get("relation"),
            )
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(relation)

        with (eval_dir / "pred_entities.json").open("w", encoding="utf-8") as f:
            json.dump(unique_entities, f, ensure_ascii=False, indent=2)
        with (eval_dir / "pred_relations.json").open("w", encoding="utf-8") as f:
            json.dump(unique_relations, f, ensure_ascii=False, indent=2)

    def _write_state(self) -> None:
        STATE_PATH.write_text(
            json.dumps(
                {
                    "last_run": datetime.utcnow().isoformat(),
                    "output_dir": str(self.output_dir),
                    "steps": self.completed_steps,
                    "resume": self.args.resume,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        INPUT_SNAPSHOT_PATH.write_text(json.dumps(self.current_snapshot, indent=2), encoding="utf-8")

    # ---- snapshot helpers -------------------------------------------------

    def _collect_input_snapshot(self) -> Dict[str, Dict]:
        files: Dict[str, Dict] = {}
        for path in TRACKED_INPUTS:
            if not path.exists():
                continue
            if path.is_file():
                files[str(path.relative_to(PROJECT_ROOT))] = self._file_meta(path)
            else:
                for file in path.rglob("*"):
                    if file.is_file():
                        files[str(file.relative_to(PROJECT_ROOT))] = self._file_meta(file)
        return {"files": files}

    @staticmethod
    def _file_meta(path: Path) -> Dict[str, float]:
        stat = path.stat()
        return {"mtime": round(stat.st_mtime, 2), "size": stat.st_size}

    def _load_previous_snapshot(self) -> Optional[Dict[str, Dict]]:
        if not INPUT_SNAPSHOT_PATH.exists():
            return None
        try:
            return json.loads(INPUT_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _diff_snapshots(
        self,
        previous: Optional[Dict[str, Dict]],
        current: Dict[str, Dict],
    ) -> Set[str]:
        if previous is None:
            return set(current["files"].keys())
        prev_files = previous.get("files", {})
        curr_files = current.get("files", {})
        changed = set()

        for path, meta in curr_files.items():
            if path not in prev_files or prev_files[path] != meta:
                changed.add(path)
        for path in prev_files:
            if path not in curr_files:
                changed.add(path)
        return changed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated pipeline runner for the Innovation Platform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for generated artefacts.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Directory for evaluation files.")
    parser.add_argument("--cache-path", default=str(PROJECT_ROOT / "embedding_vectors.json"), help="Embedding cache path.")
    parser.add_argument(
        "--query-cache-path",
        default=str(PROJECT_ROOT / "query_embeddings.json"),
        help="Cache path for query-engine embeddings.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing consolidated graph if possible.")
    parser.add_argument("--force", action="store_true", help="Force re-running every step and ignore caches.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=DEFAULT_STEPS,
        help="Subset of steps to execute (default: entire pipeline).",
    )
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualisation step.")
    parser.add_argument("--skip-export", action="store_true", help="Skip export step.")
    parser.add_argument("--with-eval", action="store_true", help="Run evaluation (requires Azure OpenAI config).")
    parser.add_argument("--auto-label", action="store_true", help="Enable LLM auto labelling during evaluation.")
    parser.add_argument("--skip-openai", action="store_true", help="Skip Azure OpenAI initialisation.")
    parser.add_argument("--sample-query", default="", help="Sample query to run after building the query engine.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of query results to print.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top nodes in analysis.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum iterations for network analysis.")
    parser.add_argument(
        "--clustering-method",
        default="hdbscan",
        choices=["hdbscan", "kmeans", "agglomerative", "spectral", "graph_threshold", "graph_kcore"],
        help="Clustering strategy for deduplication.",
    )
    parser.add_argument(
        "--clustering-kwargs",
        type=json.loads,
        default="{}",
        help='Additional clustering parameters as JSON (e.g. \'{"min_cluster_size": 3}\').',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runner = PipelineRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
