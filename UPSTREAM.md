# Upstream Boundary

This repository vendors GPT Researcher as the controlled testbed used by the
FORGE deep-research experiments.

## Upstream

- Project: GPT Researcher
- Upstream repository: `https://github.com/assafelovic/gpt-researcher`
- Upstream package version vendored here: `0.14.5`
- Exact upstream Git commit: `2131b0f` (tag `v0.14.5`, repository `assafelovic/gpt-researcher`)
- Upstream license: MIT

The package identity for this artifact is `forge-deep-research` to avoid
confusion with the official `gpt-researcher` package.

## Vendored Runtime

The following directories and files are vendored from the GPT Researcher
testbed and kept in the repository for reproducibility:

- `gpt_researcher/`
- `backend/`
- `cli.py`
- `requirements.txt`
- `pyproject.toml` dependency set, with project identity changed for this artifact
- `setup.py`, with project identity changed for this artifact

## FORGE Additions

The experiment-specific additions are:

- `forge/`: FORGE construction metadata, prompt templates, chain helpers, and retrieval exports.
- `prism/`: PRISM claim taxonomy, paper-weighted scoring, equal-weight diagnostic scoring, and CSV evaluation helpers.
- `defense/`: Root Query Anchoring helper.
- `experiments/`: experiment entry points.
- `tools/`: document-generation, ASR scoring, graph-checking, and batch helpers.
- `data/`: minimal public examples.
- `docs/`: reproducibility and scoring notes.

## Local Modifications To Vendored GPT Researcher

The release modifies the vendored runtime in these areas:

- `gpt_researcher/skills/deep_research.py`: adds `ENABLE_DEEP_RESEARCH_DEFENSE` root-query anchoring in deep-research planning and recursive query generation.
- `gpt_researcher/actions/planning_sources.py`: supports web-poison virtual URL candidates and optional candidate ranking.
- `gpt_researcher/skills/researcher.py`: supports local and web poison document paths used by the experiments.
- `gpt_researcher/config/variables/base.py`: adds experiment configuration fields.
- `gpt_researcher/config/variables/default.py`: sets experiment defaults.
- `backend/server/app.py`: adjusts research logging for experiment traces.
- `cli.py`: exposes poison document paths, deep-research parameters, and the defense switch.

The upstream `gpt_researcher` import path is intentionally preserved inside the
repository because the runtime imports itself under that Python package name.
