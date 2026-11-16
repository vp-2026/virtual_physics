# Evaluation populations

The benchmark has 1,692 goal tasks on 66 visual layouts in two tool directions. The three main models have 1,651 common completed task summaries. Their consistently prediction-reconciled main population contains 1,641 tasks per model.

The ten-task difference is caused by non-unique or missing matches between Gemini 3.1 Pro branch-summary coordinates and saved attempt-level simulator records in the retry-only or JSON-feedback branches. It is not Qwen budget exhaustion and is not a set of ten additional task failures.
Across these ten tasks there are 16 unreconciled attempt positions: 15 lack a matching non-obstructed simulator record and one has two matching records. Completion summaries remain available for the broader solve-rate controls. The main prediction-reconciled population excludes the same ten task IDs for all three main models.

| Excluded main-analysis task | Affected Gemini branch |
|---|---|
| `0356_FallingAlt_10_upward__goal_0006` | `neither` |
| `0606_Remove_8_upward__goal_0012` | `neither` |
| `0758_BalanceUnder_7_upward__goal_0014` | `neither` |
| `under29_new_0732_Falling_2_upward` | `trace_status` |
| `under29_new_0768_Prevention_5_upward` | `trace_status` |
| `under29_new_0796_FallingAlt_7_upward` | `neither` |
| `under29_new_0808_BalanceUnder_7_upward` | `neither` |
| `under29_new_0972_FallingAlt_2_upward` | `trace_status` |
| `under29_new_1205_Remove_6_upward` | `neither` |
| `under29_new_1445_Falling_6_upward` | `neither` |

Exact reported memberships are in [`results/populations.json`](../results/populations.json). Fixed analysis memberships and transfer candidates are in [`benchmark/evaluation_task_ids.json`](../benchmark/evaluation_task_ids.json). Finalized model records and counts are in [`data/finalized/manifest.json`](../data/finalized/manifest.json).
The matched novel-probe population contains 116 tasks. Robotics-ER 2 is evaluated separately on 132 canonical tasks, with its own 127-task valid novel-probe population. These denominators should not be interchanged.
The 1,641 and 1,651 counts denote common task IDs evaluated for each main model, not total episodes summed over three models.
