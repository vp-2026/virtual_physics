# Running the code

## Offline reproduction

From the repository root, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/reproduce.py --output outputs/reproduction
```

This reads the released records and writes scores, tables, figures, random-search and no-motion baselines to `outputs/reproduction/`. It makes no model API calls. Use a new output directory for each run; generated files stay local.

- [`evaluation/`](evaluation/) contains scoring and statistical estimators.
- [`release/`](release/) contains dataset, geometry, and numerical checks.
- [`render_figures.py`](render_figures.py) and [`write_results.py`](write_results.py) generate figures and tables.

## Experiment entry points

The [prompt protocol](../prompts/README.md) defines the feedback conditions, response format, and stopping rules.

```bash
python scripts/experiment/forked_feedback_runner.py --help
python scripts/experiment/transfer_sidecar_runner.py --help
```

[`experiment/forked_feedback_runner.py`](experiment/forked_feedback_runner.py) implements action selection, chosen-action prediction, and feedback controls. [`experiment/transfer_sidecar_runner.py`](experiment/transfer_sidecar_runner.py) implements the isolated early and terminal novel-action probes. Both use [`experiment/goal_probe_runner.py`](experiment/goal_probe_runner.py).

Collection dependencies are listed in [`requirements.txt`](../requirements.txt). Exact collected model settings are retained in the corresponding model records.
