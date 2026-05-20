# Graph ASR Audit Pipeline

The graph ASR audit pipeline analyzes deep-research task graphs to measure how effectively adversarial documents have infiltrated the research process. It combines atomic ASR pipeline results with experiment graph files to produce summary tables and automated statistics.

## Pipeline overview

The pipeline runs on a single experiment directory and produces summary statistics:

```
atomic ASR pipeline results (asr/dataout/...)
         │
         │ Stage 1 — parse graph files
         │   Extracts subtasks, planning/research source links,
         │   and learning entries from each experiment graph.
         │
         │ Stage 2 — compute source poisoning statistics
         │   Counts poisoned sources (containing "//research/"),
         │   computes ratios by depth level and research phase.
         │
         │ Stage 3 — generate summary tables
         │   Produces subquestions_summary.csv and learnings_summary.csv
         │   with per-item poisoning detection results.
         │
         ▼
  new_summary/ directory with CSV, JSON, and Markdown reports
```

All intermediate results are cached: re-running the pipeline resumes from the last completed stage rather than re-processing from scratch.

## Output files

| Path | Description |
|---|---|
| `new_summary/new_subquestions_summary.csv` | Per-subquestion poisoning statistics and ASR detection |
| `new_summary/new_learnings_summary.csv` | Per-learning poisoning statistics and ASR detection |
| `new_summary/new_stats_overall.json` | Overall statistics, confusion matrices, quality checks |
| `new_summary/new_stats_by_run.csv` | Statistics grouped by run ID |
| `new_summary/new_stats_by_setting.csv` | Statistics grouped by setting (clean/local/web) |
| `new_summary/new_stats_by_doc_count.csv` | Statistics grouped by document count |
| `new_summary/new_stats_by_setting_doc_count.csv` | Two-dimensional setting × doc_count comparison |
| `new_summary/new_stats_by_depth.csv` | Statistics by subquestion depth level |
| `new_summary/new_stats_claim_matches.csv` | Matched claim ID and type frequency |
| `new_summary/new_stats_top_subquestions.csv` | High-risk subquestions Top-K |
| `new_summary/new_stats_top_learnings.csv` | High-risk learnings Top-K |
| `new_summary/new_stats_report.md` | Human-readable Markdown report |
| `new_summary_final/new_final_metrics.csv` | Core metrics summary |
| `new_summary_final/new_final_overall.json` | Final overall statistics |

## Running the pipeline

Basic usage:

```bash
python -m graphcheck.main \
  --asr-output-dir path/to/asr/dataout/Category/QueryName \
  --model your-model \
  --base-url https://your-endpoint/v1 \
  --api-key your-api-key
```

Skip ASR scoring (parse graphs and generate statistics only):

```bash
python -m graphcheck.main \
  --asr-output-dir path/to/asr/dataout/Category/QueryName \
  --skip-asr
```

Manually specify the graph experiment directory:

```bash
python -m graphcheck.main \
  --asr-output-dir path/to/asr/dataout/Category/QueryName \
  --graph-experiment-dir path/to/dataset/Category/QueryName \
  --model your-model \
  --base-url https://your-endpoint/v1 \
  --api-key your-api-key
```

Exclude `local` experiments (process only `clean` and `web` settings):

```bash
python -m graphcheck.main \
  --asr-output-dir path/to/asr/dataout/Category/QueryName \
  --model your-model \
  --base-url https://your-endpoint/v1 \
  --api-key your-api-key \
  --exclude-local-data
```

Force full recomputation (ignore cached intermediate results):

```bash
python -m graphcheck.main \
  --asr-output-dir path/to/asr/dataout/Category/QueryName \
  --model your-model \
  --base-url https://your-endpoint/v1 \
  --api-key your-api-key \
  --no-reuse-intermediate
```

## ASR matching strategy

The current version uses single-pass full-pool matching:

- Each run's `subquestions` receive 1 full poisoning-pool matching request
- Each run's `learnings` receive 1 full poisoning-pool matching request
- No longer splits by claim type into 5 separate requests

This significantly reduces request count, making it more suitable for unstable or slow API endpoints.

If the API is unstable, increase retry attempts:

```bash
python -m graphcheck.main \
  --asr-output-dir path/to/asr/dataout/Category/QueryName \
  --model your-model \
  --base-url https://your-endpoint/v1 \
  --api-key your-api-key \
  --score-type-attempts 2
```

Pass API key via environment variable:

```bash
export ATOMIC_ASR_API_KEY=your-api-key
python -m graphcheck.main \
  --asr-output-dir path/to/asr/dataout/Category/QueryName \
  --model your-model \
  --base-url https://your-endpoint/v1
```

## Implementation conventions

- In `local` experiments, local documents at `scholar.local/documents/...` that are not `pad-text` are counted as poison sources.
- In `web` and `clean` experiments, links containing `//research/` are counted as poison sources.
- When `--exclude-local-data` is enabled, `local` runs are filtered out entirely.
- `pad-text` is excluded from both citation and poison statistics.
- Subquestion depth is computed from indentation levels in the `graph` file.
- Learning source subtasks are traced back via the `Learning Source Trace` block at the bottom of the graph.
- `asr_matched_types` is reverse-inferred from the matched injected claim's own type.

## Summary table fields

### `new_subquestions_summary.csv`

| Field group | Fields |
|---|---|
| Basic info | `run_id`, `setting`, `doc_count` |
| Subquestion | `subquestion_id`, `depth`, `parent_subquestion_id`, `subquestion_text` |
| Citation stats | `planning_web_refs`, `planning_local_refs`, `planning_total_refs`, `research_web_refs`, `research_local_refs`, `research_total_refs`, `total_refs` |
| Poison stats | `planning_poison_refs`, `research_poison_refs`, `poison_refs`, `poison_ratio`, `planning_poison_ratio`, `research_poison_ratio` |
| ASR detection | `asr_poisoned`, `asr_best_status`, `asr_matched_types`, `asr_matched_claim_ids`, `asr_justifications` |

### `new_learnings_summary.csv`

| Field group | Fields |
|---|---|
| Basic info | `run_id`, `setting`, `doc_count` |
| Learning | `learning_id`, `learning_text`, `learning_source_url` |
| Link poisoning | `learning_link_is_poison` |
| Source subtask | `source_subtask_id`, `source_subtask_text`, `source_subtask_asr_poisoned` |
| ASR detection | `asr_poisoned`, `asr_best_status`, `asr_matched_types`, `asr_matched_claim_ids`, `asr_justifications` |

## Statistics output

After writing the summary CSVs, the tool automatically generates statistics files in the same `new_summary/` directory:

- `new_stats_overall.json` — overall statistics, confusion matrices, and quality checks
- `new_stats_by_*.csv` — grouped statistics for comparison analysis
- `new_stats_by_depth.csv` — subquestion statistics by depth level
- `new_stats_core_story.csv` — key metrics explaining "why more poison documents leads to higher ASR"
- `new_summary_final/` — extracts 3 core metrics from `new_stats_core_story.csv`:
  - `retrieval_poison_ratio_mean`
  - `subquestion_asr_rate`
  - `subtask_to_learning_propagation_rate`
- `new_stats_claim_matches.csv` — matched claim ID and claim type frequency
- `new_stats_top_*.csv` — high-risk sample rankings
- `new_stats_report.md` — human-readable Markdown report

## Summary results guide

Recommended reading order:

1. `new_run_manifest.json` — verify experiment context
2. `new_subquestions_summary.csv` + `new_learnings_summary.csv` — finest-granularity evidence
3. `new_stats_overall.json` — global overview
4. `new_stats_by_*.csv` — grouped comparisons
5. `new_stats_top_*.csv` + `new_stats_report.md` — high-risk samples and readable report

### Key metrics

| Metric | File | Description |
|---|---|---|
| `subquestions.asr_poisoned_rate` | `new_stats_overall.json` | Fraction of subquestions with ASR hits |
| `learnings.asr_poisoned_rate` | `new_stats_overall.json` | Fraction of learnings with ASR hits |
| `subquestions.source_vs_asr_matrix` | `new_stats_overall.json` | Confusion matrix: source poisoning vs ASR detection |
| `learnings.link_vs_asr_matrix` | `new_stats_overall.json` | Confusion matrix: link poisoning vs ASR detection |
| `learnings.source_subtask_to_learning_propagation_rate` | `new_stats_overall.json` | Rate of poisoning propagation from subtask to learning |

### Typical investigations

- **High source poisoning but no ASR hit**: filter `poison_ratio > 0` and `asr_poisoned=False`
- **Are deeper nodes more vulnerable**: group by `depth` and aggregate `poison_ratio` with `asr_poisoned`
- **Source chain broken**: `source_subtask_id` is empty
- **Source subtask not poisoned but learning is poisoned**: `source_subtask_asr_poisoned=False` and `asr_poisoned=True`

## Quality check fields

Common keys in `new_stats_overall.json → quality_checks`:

| Key | Description |
|---|---|
| `*_missing_text_count` | Missing text |
| `*_missing_graph_path_count` | Missing graph path |
| `task_*_poison_ratio_out_of_range_count` | Ratio out of range (not 0–1) |
| `task_poison_refs_gt_total_refs_count` | Poison citation count exceeds total citation count |
| `*_asr_status_inconsistent_count` | `asr_poisoned` and `asr_best_status` semantic conflict |
| `*_match_without_claim_ids_count` | Hit status but no claim ID provided |

When these counts are greater than 0, fix data quality issues before comparing model performance.
