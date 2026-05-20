from __future__ import annotations

from pathlib import Path
import json
import re

from .models import ExperimentContext, RunArtifacts


DATASET_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "HistoricalDevelopment": ("HistoricalDevelopment",),
    "FactualSurveys": ("FactualSurveys",),
    "CausalExplanationOrTrendJudgment": ("CausalExplanationOrTrendJudgment",),
    "MethodComparison": ("MethodComparison",),
    "ControversialTopics": ("ControversialTopics",),
}


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.replace('"', " ").replace("'", " ")
    text = text.replace("“", " ").replace("”", " ").replace("’", " ")
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    return text


def _infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _query_fragment_from_dir_name(dir_name: str) -> str:
    return re.sub(r"^\d+_", "", dir_name).strip()


def _query_number_from_query_id(query_id: str) -> str:
    match = re.search(r"(\d+)$", query_id)
    if not match:
        raise ValueError(f"Could not infer numeric prefix from query_id={query_id!r}")
    return f"{int(match.group(1)):02d}"


def _run_numeric_token(run_id: str) -> str | None:
    matches = re.findall(r"(\d+)", run_id)
    if not matches:
        return None
    return matches[-1]


def _query_number_from_experiment_dir(experiment_dir: Path) -> str | None:
    match = re.search(r"(?:^|[^0-9])q(\d{2})(?:[^0-9]|$)", experiment_dir.name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _resolve_dataset_category_dir(repo_root: Path, category_name: str) -> Path:
    dataset_root = repo_root / "dataset"
    aliases = DATASET_CATEGORY_ALIASES.get(category_name, (category_name,))
    for alias in aliases:
        candidate = dataset_root / alias
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not map asr category {category_name!r} to a dataset category directory under {dataset_root}"
    )


def _resolve_dataset_experiment_dir(
    dataset_category_dir: Path,
    query_id: str,
    query_fragment: str,
) -> Path:
    prefix = f"{_query_number_from_query_id(query_id)}_"
    normalized_fragment = _normalize(query_fragment)
    candidates = [
        path
        for path in dataset_category_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset experiment directory under {dataset_category_dir} starts with {prefix!r}"
        )

    if normalized_fragment:
        exact_like = [
            path
            for path in candidates
            if _normalize(path.name).startswith(_normalize(prefix + query_fragment))
            or normalized_fragment in _normalize(path.name)
            or _normalize(path.name).startswith(_normalize(prefix) + normalized_fragment)
        ]
        if len(exact_like) == 1:
            return exact_like[0]
        if exact_like:
            exact_prefix = [
                path for path in exact_like if _normalize(path.name).startswith(_normalize(prefix) + normalized_fragment)
            ]
            if len(exact_prefix) == 1:
                return exact_prefix[0]

    if len(candidates) == 1:
        return candidates[0]

    candidate_names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"Could not uniquely resolve the dataset experiment for query_id={query_id!r}, "
        f"fragment={query_fragment!r}. Candidates: {candidate_names}"
    )


def _resolve_graph_path(experiment_dir: Path, run_id: str) -> Path:
    exact_matches = sorted(experiment_dir.rglob(f"{run_id}_graph.md"))
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise FileNotFoundError(
            f"Multiple exact graph files matched run_id={run_id!r} under {experiment_dir}: "
            + ", ".join(str(path) for path in exact_matches)
        )

    pattern_matches = sorted(experiment_dir.rglob(f"{run_id}_*_graph.md"))
    if len(pattern_matches) == 1:
        return pattern_matches[0]
    if len(pattern_matches) > 1:
        raise FileNotFoundError(
            f"Multiple graph files matched run_id={run_id!r} under {experiment_dir}: "
            + ", ".join(str(path) for path in pattern_matches)
        )

    run_numeric_token = _run_numeric_token(run_id)
    query_number = _query_number_from_experiment_dir(experiment_dir)
    if run_numeric_token:
        numeric_matches = [
            path
            for path in sorted(experiment_dir.rglob("*_graph.md"))
            if run_numeric_token in path.stem
        ]
        if query_number:
            query_numeric_matches = [
                path for path in numeric_matches if f"_{query_number}_" in path.stem
            ]
            if len(query_numeric_matches) == 1:
                return query_numeric_matches[0]
            if len(query_numeric_matches) > 1:
                raise FileNotFoundError(
                    f"Multiple graph files matched run_id={run_id!r} via query number {query_number!r} under {experiment_dir}: "
                    + ", ".join(str(path) for path in query_numeric_matches)
                )
        if len(numeric_matches) == 1:
            return numeric_matches[0]
        if len(numeric_matches) > 1:
            raise FileNotFoundError(
                f"Multiple graph files matched run_id={run_id!r} via numeric token {run_numeric_token!r} under {experiment_dir}: "
                + ", ".join(str(path) for path in numeric_matches)
            )

    folder_matches = [path for path in experiment_dir.rglob("*") if path.is_dir() and path.name == run_id]
    for folder in folder_matches:
        candidate = folder / "task_graph.md"
        if candidate.exists():
            return candidate

    available_graph_files = sorted(experiment_dir.rglob("*_graph.md"))
    available_run_ids: list[str] = []
    for graph_file in available_graph_files:
        stem = graph_file.stem
        if stem.endswith("_graph"):
            available_run_ids.append(stem[: -len("_graph")])
    sample_run_ids = ", ".join(available_run_ids[:8]) if available_run_ids else "(none)"

    raise FileNotFoundError(
        f"Could not resolve a graph file for run_id={run_id!r} under {experiment_dir}. "
        f"This often means --asr-output-dir and --graph-experiment-dir are from different queries/categories. "
        f"Sample available graph run_ids here: {sample_run_ids}"
    )


def resolve_experiment_context(
    asr_output_dir: Path,
    graph_experiment_dir: Path | None = None,
) -> ExperimentContext:
    repo_root = _infer_repo_root()
    asr_output_dir = asr_output_dir.resolve()
    if not asr_output_dir.exists():
        raise FileNotFoundError(f"ASR output directory does not exist: {asr_output_dir}")

    queries_path = asr_output_dir / "data" / "queries.json"
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries.json under: {queries_path}")

    queries_payload = _read_json(queries_path)
    if not isinstance(queries_payload, list) or not queries_payload:
        raise ValueError(f"Expected a non-empty query list in {queries_path}")

    query_record = queries_payload[0]
    query_id = str(query_record["query_id"]).strip()
    query_text_fragment = str(query_record.get("query_text") or "").strip()
    if not query_text_fragment:
        query_text_fragment = _query_fragment_from_dir_name(asr_output_dir.name)

    category_name = asr_output_dir.parent.name
    if graph_experiment_dir is not None:
        dataset_experiment_dir = graph_experiment_dir.resolve()
        if not dataset_experiment_dir.exists() or not dataset_experiment_dir.is_dir():
            raise FileNotFoundError(
                f"The provided graph experiment directory does not exist or is not a directory: {dataset_experiment_dir}"
            )
    else:
        dataset_category_dir = _resolve_dataset_category_dir(repo_root, category_name)
        dataset_experiment_dir = _resolve_dataset_experiment_dir(
            dataset_category_dir=dataset_category_dir,
            query_id=query_id,
            query_fragment=query_text_fragment,
        )
    query_text = _query_fragment_from_dir_name(dataset_experiment_dir.name)

    report_dir = asr_output_dir / "data" / "reports" / query_id
    if not report_dir.exists():
        raise FileNotFoundError(f"Missing prepared report directory: {report_dir}")

    runs: list[RunArtifacts] = []
    for report_json_path in sorted(report_dir.glob("*.json")):
        payload = _read_json(report_json_path)
        run_id = str(payload["run_id"]).strip()
        setting = str(payload["setting"]).strip()
        doc_count = int(payload["doc_count"])
        graph_path = _resolve_graph_path(dataset_experiment_dir, run_id)
        runs.append(
            RunArtifacts(
                run_id=run_id,
                setting=setting,
                doc_count=doc_count,
                report_json_path=report_json_path,
                graph_path=graph_path,
            )
        )

    new_parsed_dir = asr_output_dir / "new_parsed"
    new_scoring_dir = asr_output_dir / "new_scoring"
    new_traces_dir = asr_output_dir / "new_traces"
    new_summary_dir = asr_output_dir / "new_summary"
    for directory in (new_parsed_dir, new_scoring_dir, new_traces_dir, new_summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ExperimentContext(
        asr_output_dir=asr_output_dir,
        dataset_experiment_dir=dataset_experiment_dir,
        category_name=category_name,
        query_id=query_id,
        query_text=query_text,
        query_text_fragment=query_text_fragment,
        runs=runs,
        new_parsed_dir=new_parsed_dir,
        new_scoring_dir=new_scoring_dir,
        new_traces_dir=new_traces_dir,
        new_summary_dir=new_summary_dir,
    )
