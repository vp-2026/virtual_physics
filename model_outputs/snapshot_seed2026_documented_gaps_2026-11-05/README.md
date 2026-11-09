# Closed seed-2026 archive with documented gaps

This snapshot includes every strictly terminal episode available after the
bounded recovery passes. Collection is closed; incomplete histories remain
private and are identified in the missingness manifest. Earlier snapshots
and the frozen July 27 analysis tables are preserved unchanged.

| Model | Complete main episodes | Missing main | Complete canonical sidecars |
|---|---:|---:|---:|
| GPT-5 | 1,692/1,692 | 0 | 132/132 |
| Gemini 3.1 Pro Preview | 1,691/1,692 | 1 | 132/132 |
| Qwen 3.6 Plus | 1,652/1,692 | 40 | 131/132 |

There are 5,035 main episodes and 395 canonical sidecar bundles. All three
models have 132 canonical main episodes. A main episode contains the three
primary conditions (`full`, `neither`, `trace_status`) for one model, goal,
and seed. Some historical records also contain `status_only`; its separate
coverage is reported. A completed sidecar bundle may still require exclusion
from a particular transfer analysis under clean-probe rules.

## Files and interpretation

- `data/`: compressed JSONL records, one completed episode per line.
- `snapshot_manifest.json`: shard counts, sizes, SHA-256 hashes, and recovery
  policies; identifies the 113 newly included Qwen recovery replicas.
- `coverage.csv`: expected and complete denominators by model and condition.
- `missing_units.csv` and `missing_units.json`: all 41 incomplete main
  episodes and the one incomplete Qwen sidecar, with benchmark indices and
  recorded stopping reasons. These are not experimental failures.
- `analysis_populations.json`: complete and matched index sets. Availability
  is not a substitute for clean-probe filtering in transfer analyses.
- `planning_available.csv`: solve rates among completed episodes, plus
  worst-case sensitivity bounds over the entire expected 1,692-goal suite.
  Bounds are not confidence intervals and do not impute failure outcomes.
- `recovery_exceptions.csv`: all 113 added Qwen replicas, identifying recorded
  format-repair calls and requests with expanded output allowances.
- `frozen_analysis_manifest.json`: hashes and row counts for the historical
  paper-analysis tables, which are not recomputed from this later archive.

Missingness may depend on scene, model, output length, placement validity, or
provider behavior. Treating the available subset as a random sample is not
justified. Comparisons must identify their matched population and disclose
recovery budgets. Prediction and transfer scores require their own valid,
matched observations and denominators.

## Recovery provenance

The recovery selected previously nonterminal units without regard to their
partial successes. All 113 terminal Qwen recovery replicas are included;
no already terminal historical episode was replaced. Original partial
histories are retained privately. The `recovery_provenance` field separates
these later replicas from the frozen paper-analysis population.

Qwen's empty, length-truncated final answers alone could trigger doubling of
the output allowance, up to 32,768 tokens and at most two additional calls.
Exact inputs were preserved, and the first nonempty answer was retained
regardless of correctness. Available recovery-call records, including the
truncated responses and allowances, accompany the affected released units.
Ordinary-budget and expanded-budget observations should not be presented as
having equal inference budgets. `recovery_exceptions.csv` distinguishes a
fresh recovery replica from actual evidence of token expansion. Its counts
are log records, not billed-request counts. Recorded logical-call wall time
can include retries and pauses within a call, excludes other collection and
recovery histories, and is not end-to-end experiment time. A zero maximum
means no token-recovery log, not a zero-token request. Do not add the separate
recovery-log durations to the logical-call duration, which may include them.

Gemini's final episode exhausted cumulative placement-repair ceilings of 24,
48, and 96 while retaining four valid trace/status attempts. It remains
incomplete and excluded from completed outputs. Its histories and cumulative
counters were preserved. The eight-valid-attempt limit did not change.
Provider content-inspection rejections were not bypassed. The exact gap breakdown is recorded in `missing_units.json`.

## Validation and exclusions

From the repository root, run:

```bash
python3 scripts/release/validate_documented_snapshot.py
```

The validator checks shard hashes, decompressed and plain contents, strict
terminal summaries, cross-model task identity, completed sidecar bundles,
complete index accounting including gaps, matched populations, planning
numerators/denominators, preserved older records, and frozen table hashes.
The export excludes goal-switch transfer materials, incomplete sidecar
bundles, credentials, personal paths and identifiers, and regenerable media
and full simulator traces. It contains prompts, responses, parsed actions,
predictions, simulator endpoint truth, and available usage/latency metadata.

This is the original three-model archive. The separately released Robotics ER2
extension and original ER1.6 control each have their own dated directory. Use
the population required by each estimator; archive coverage alone does not
determine the denominator of a paper result.
