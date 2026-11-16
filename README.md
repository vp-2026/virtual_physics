# Virtual Physics: planning, prediction, and transfer

A joint evaluation of whether multimodal models can choose successful actions, predict their consequences, and use interaction to predict new interventions in the same physical environment.

## Finalized benchmark and results

- **Benchmark:** 1,692 goals on 66 layouts, each with upward and downward tools. [`benchmark/`](benchmark/) contains the default task definitions and simulator-aligned assets.
- **Models:** GPT-5, Gemini 3.1 Pro, and Qwen 3.6 Plus; Gemini Robotics-ER 2 is a separate 132-task canonical extension.
- **Data:** [`data/finalized/manifest.json`](data/finalized/manifest.json) identifies the authoritative model records and their provenance. Human participant-level records are not distributed.
- **Results:** [`results/RESULTS.md`](results/RESULTS.md) is the numerical reference, including exact analysis populations, confidence intervals, feedback controls, prediction correspondence, and corrected-scene VQA. Figures and appendix tables are generated from these records.

On the finalized 1,641-task main population, full-feedback solve rates are 41.80% (GPT-5), 46.50% (Gemini), and 41.62% (Qwen). Feedback improves later success beyond retrying for all three; prediction benefits vary by model. Successful action has a modest adjusted association with chosen-action prediction, while relationships involving novel-action prediction remain uncertain after difficulty and exposure controls. The matched transfer population contains 116 tasks; Robotics-ER 2 is reported separately.

Action success is binary. Prediction quality gives graded spatial credit to object-endpoint predictions; it is not percentage accuracy. Feedback benefits are compared against retry controls, and correspondence analyses account for task difficulty and feedback exposure. See the results reference for model-specific estimates and uncertainty.

The [reference baselines](results/baselines/README.md) compare success with uniform random placement and prediction quality with a no-motion forecast. See the [prompt protocol](prompts/README.md), [worked two-attempt example](docs/worked_interaction_example.md), and [evaluation populations](docs/evaluation_populations.md) for concrete definitions.

## Reproduce offline

Python 3.12 is the validated runtime. From this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/reproduce.py --output outputs/reproduction
```

This command checks dataset integrity, verifies all 66 scenes against the simulator, re-scores recorded responses, recomputes tables and bootstrap intervals, independently checks endpoint scores and Figure 4 estimators, calculates random-placement and no-motion baselines, and renders vector Figures 3 and 4. It makes **no model API calls**. Use a new output directory for each reproduction.

For integrity checks alone:

```bash
python scripts/release/verify_release.py --geometry
```

The output includes `summary.json`, `estimates.csv`, exact task IDs in `populations.json`, `tables/`, `baselines/`, and `figures/plotted_values.json`. Collection scripts are in [`scripts/experiment/`](scripts/experiment/); the offline analysis requires no credentials.

## Data and licensing

Three inconsistent initial screenshots were corrected, and affected model episodes and fixed probes were recollected. Unaffected observations remain unchanged. The model-data manifest records source fingerprints and replacements.

Project and simulator code use the [MIT License](LICENSE), with attribution in [NOTICE.md](NOTICE.md). See [LICENSE_STATUS.md](LICENSE_STATUS.md) for third-party and task-asset scope.
