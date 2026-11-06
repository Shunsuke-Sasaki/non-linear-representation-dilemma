# Experiment 0485: shared-mapping algorithm pilot

Issue: https://github.com/llm-jp/experiments/issues/485

Pilot for transferring algorithm correspondence across models with a shared
non-linear mapping (RevNet) and per-model orthogonal alignment, on the
distributive-law task (E = AndOrAnd, F = AndOr; 8 conditions EEE..FFF).

This directory mirrors the experiment directory on mdx
(`/model/experiments/0485_shared-mapping-algo-pilot`):

- `code/pilot.py`, `code/test_pilot.py`: pilot runner and unit tests
- `code/pilot.sbatch`, `code/pilot-smoke.sbatch`: Slurm job scripts (mdx llm-jp)
- `config/pilot_config.json`: all sizes, seeds and hyper-parameters
- `environment/requirements-pilot.{txt,lock}`: Python environment (Python 3.12.13)

`pilot.py` imports `das` from `code/source/das/`. On mdx that directory is a copy
of the top-level `das/` of this repository at commit `88d6523`
(`source_commit` in the config). To run from a fresh clone, copy `das/` to
`code/source/das/` first.
