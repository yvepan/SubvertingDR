# Open Source Layout

This repository is organized as a self-contained experiment release. The GPT
Researcher runtime used by the experiments is included directly in the
repository, so no separate checkout is required.

## Public Modules

- `cli.py`: command line entry point for deep research reports.
- `gpt_researcher/`: GPT Researcher runtime used by this release.
- `backend/`: report writers, server utilities, and report-type helpers.
- `forge/`: FORGE method scaffolds: chain metadata helpers, and the Appendix B prompt templates (Steps 1, 2a, 2b) used for adversarial document construction plus the intra/inter-document review prompts.
- `prism/`: PRISM-style claim taxonomy and metric aggregation helpers.
- `experiments/`: stable command entry points for local, network, and depth experiments.
- `defense/`: Root Query Anchoring helper used by deep-research defense.
- `tools/`: release-local helpers for document generation, ASR scoring, depth experiments, and graph checks.
- `data/queries.example.json`: one minimal query record showing the expected data schema.
- `data/claims.example.csv`: one minimal claim-level ASR scoring example.
- `docs/asr_scoring.md`: ASR formula, input schema, and command-line scoring usage.
- `UPSTREAM.md`: vendored GPT Researcher attribution and local modification boundary.

## Private Or Generated Inputs

The following should stay outside git:

- Full adversarial document sets.
- Raw experiment outputs and ASR outputs.
- API keys, provider endpoints, and machine-specific absolute paths.

## Main Experiment Commands

Generate a document with defense enabled:

```bash
python -m tools.generate_document "Your query" --report-source hybrid --web-poison-dir ./web-poison-docs --enable-defense
```

Generate a document with defense disabled:

```bash
python -m tools.generate_document "Your query" --report-source hybrid --web-poison-dir ./web-poison-docs --disable-defense
```

Compute ASR from normalized claim annotations:

```bash
python -m tools.score_claim_csv data/claims.example.csv
```

Network/web-poison depth sweep:

```bash
python -m experiments.run_depth
```

Local or hybrid run with supplied documents:

```bash
python -m experiments.run_local "Your query" --web-poison-dir ./web-poison-docs --breadth 3 --depth 3
```

## Reproducibility Notes

Hybrid web-poison experiments must pass an explicit empty local-poison
directory. The experiment runners use `outputs/empty_local_poison_docs` for
this, which prevents default `local-poison-docs` files from contaminating
web-only measurements.

URL reranking is implemented as:

```text
score = bm25_weight * normalized_bm25 + embedding_weight * normalized_embedding_similarity
```

The current default is `bm25_weight=0.4` and `embedding_weight=0.6`.
