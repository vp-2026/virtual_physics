# Collection status and recovery limits

Status observed November 3, 2026 (Pacific time). This is a documentation update,
not a new model-output release. The published snapshot and accepted-paper
analysis tables retain their existing populations and hashes.

## Published coverage and later collection

| Original model | Main episodes in the published August 2 snapshot | Main episodes reported by later local checkpoints | Canonical sidecars |
|---|---:|---:|---:|
| GPT-5 | 1,692/1,692 | 1,692/1,692 | 132/132 |
| Gemini 3.1 Pro Preview | 1,691/1,692 | 1,691/1,692 | 132/132 |
| Qwen 3.6 Plus | 1,539/1,692 | 1,629/1,692 | 131/132 |

The later Qwen count is a dated collection checkpoint, not a claim that those
additional outputs have been published or passed the final consolidated
release audit. Recovery remains in progress for eligible Qwen units. The
sidecar counts shown above are the completed bundles available at this check.
Partial episodes and partial sidecars are incomplete observations. They must
not be counted as unsuccessful puzzle-solving trials or assigned failure
scores in action, prediction, or transfer analyses.

## Gemini: final episode remains incomplete

The final Gemini episode was retried under documented procedural-repair
ceilings of 24, then 48, then 96. The last continuation preserved the first
48 blocked proposals and added at most 48 more; it did not reset the repair
counter or restart the episode.

The trace/status branch reached 96 cumulative blocked placement proposals
while trying to obtain valid attempt 5. It retains four valid attempts and
remains incomplete. The latest checkpoint records 96 blocked proposals and
zero duplicate proposals for this attempt. No further retries are running
for this episode under the exhausted limit. The cap of eight valid attempts
and the existing output-token allowances were unchanged.

The higher repair budget is an inference-budget exception. It must be
identified separately in any future analysis, rather than silently folded
into the frozen paper results. Prior attempts, proposal histories, and the
earlier recovery authorizations are retained.

## Qwen: ongoing bounded recovery

Qwen recovery uses local workers with the OpenRouter API. These calls do not
require a Google Cloud compute instance. Requests whose final answer was
empty because of length truncation may use the separately authorized output
allowance expansion up to 32,768 tokens. The exact input is preserved and the
first nonempty answer is retained regardless of correctness. These exceptions
also require explicit inference-budget provenance.

Some main episodes and one canonical sidecar remain blocked by provider
content inspection. Those rejections are not bypassed. Exhausted procedural
limits, interrupted runs, and provider rejections remain incomplete data;
none establish benchmark failure outcomes.

## Release boundary

The repository does not claim complete 1,692-episode coverage for Gemini or
Qwen, or complete Qwen sidecar coverage. A later complete archive requires
terminal-state and index coverage checks, sidecar validation, shard hashes,
analysis-coverage checks, and de-identification of plain and compressed files.
Goal-A-to-Goal-B materials remain excluded. This note changes no output shard
or accepted-paper analysis table.
