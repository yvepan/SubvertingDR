# Subverting Deep Research

Code release for the FORGE attack and PRISM evaluation framework.
The experiment-specific code lives in `forge/`, `prism/`, `defense/`, `experiments/`, and `tools/`.
GPT Researcher is vendored directly so the artifact reproduces without a conflicting upstream install.

---

## Repository Layout

| Path | Description |
|---|---|
| `cli.py` | Command-line entry point for deep-research report generation |
| `gpt_researcher/` | Vendored GPT Researcher runtime (v0.14.5, modified — see `UPSTREAM.md`) |
| `backend/` | Report writers and server utilities used by `cli.py` |
| `forge/` | FORGE scaffolds: chain metadata helpers and Appendix B prompt templates (Steps 1, 2a, 2b) |
| `prism/` | PRISM taxonomy, weighted scoring, and the full atomic ASR evaluation pipeline |
| `defense/` | Root Query Anchoring (RQA) defense helper |
| `experiments/` | Stable entry points for depth-sweep and network-condition runs |
| `tools/` | Document generation, ASR scoring, graph checks, and batch experiment helpers |
| `data/` | Schema examples (`queries.example.json`, `claims.example.csv`) |
| `docs/` | Implementation notes, ASR scoring guide, and web-poison source-selection details |
| `UPSTREAM.md` | GPT Researcher attribution and local modification inventory |

---

## Setup

**Requirements:** Python 3.11+, LLM/embedding API credentials, optionally a private experiment dataset.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials, or export them directly:

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=https://api.openai.com/v1
export RETRIEVER=duckduckgo
```

To reproduce the full batch experiments, point the runners at your dataset:

```bash
export FORGE_DATASET_ROOT=/path/to/poison-dataset-root   # defaults to data/
export EXPERIMENT_OUTPUTS_ROOT=/path/to/outputs           # defaults to outputs/
```

---

## Function 1 — Generate Research Reports

### With the defense wrapper

```bash
python -m tools.generate_document "Your query" \
  --report-source hybrid \
  --web-poison-dir /path/to/web-poison-docs \
  --enable-defense
```

```bash
python -m tools.generate_document "Your query" \
  --report-source hybrid \
  --web-poison-dir /path/to/web-poison-docs \
  --disable-defense
```

`--enable-defense` activates Root Query Anchoring inside the deep-research planner;
`--disable-defense` turns it off.

### Directly via cli.py

```bash
python cli.py "Your query" \
  --report_type deep \
  --report_source web \
  --tone objective \
  --enable-defense \
  --no-pdf \
  --no-docx
```

### Batch experiments

```bash
# Depth-ablation sweep (δ ∈ {1, 2, 3, 4}) — paper Figure 3
python -m experiments.run_depth

# Network-condition runs (fixed δ = 2, varying j) — paper Figure 2
python -m experiments.run_network
```

### Graph integrity check

```bash
python tools/checkgraph.py outputs/task_xxx_graph.md
```

---

## Function 2 — Compute PRISM / ASR

### Scoring formula

Per-type ASR:

```
ASR_t = infected_claims_t / total_claims_t
```

Paper PRISM score (weighted infected claim mass):

```
PRISM = Σ_t  weight(t) · infected_t  /  Σ_t  weight(t) · total_t
```

| Claim type | Weight |
|---|---|
| factual | 4 |
| prescriptive | 5 |
| evaluative | 6 |
| causal | 7 |
| framing | 8 |

### Running the scorer

```bash
# Paper-weighted PRISM score
python -m tools.score_claim_csv data/claims.example.csv

# Equal-weight diagnostic ASR
python -m tools.score_claim_csv data/claims.example.csv --weighting equal

# Grouped output per report, CSV format
python -m tools.score_claim_csv data/claims.example.csv --group-by report_id --format csv
```

See [`docs/asr_scoring.md`](docs/asr_scoring.md) for the full input schema and interpretation.

### Running the atomic ASR pipeline

The `prism/` module ships the full three-stage LLM pipeline used in the paper:

```bash
python -m prism.run_pipeline \
  --experiment-dir /path/to/experiment \
  --output-dir     /path/to/outputs \
  --model          gemini-3.1-flash-lite \
  --base-url       https://generativelanguage.googleapis.com/v1beta/openai \
  --api-key-env    GOOGLE_API_KEY
```

See [`prism/README.md`](prism/README.md) for the full pipeline guide, input layout, and output schema.

---

## Experimental Settings

The web-poison runs in the paper use:

| Parameter | Value |
|---|---|
| `--report_type` | `deep` |
| `--report_source` | `hybrid` |
| `--tone` | `objective` |
| BM25 blend weight (α) | 0.4 |
| Embedding blend weight (1−α) | 0.6 |
| Embedding model | `text-embedding-3-small` |

Hybrid runs always pass an explicit empty local-poison directory
(`outputs/empty_local_poison_docs`) so that local documents do not
contaminate web-only measurements.

---

## Release Boundary

This repository contains the code needed to reproduce the released experiments.
It does **not** include full adversarial document sets, raw experiment outputs,
evaluator outputs, or API credentials.
The `data/` directory holds minimal schema examples only;
set `FORGE_DATASET_ROOT` to point at the private dataset for full batch reproduction.
