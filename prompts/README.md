# Exact prompting protocol

The two system-prompt files are exact text used by the runner. The only
difference between `system_full.txt` and `system_compact.txt` is the first role
sentence:

- full: explicitly identifies deterministic 2-D rigid-body physics;
- compact: identifies only a visual tool-placement task.

Everything else—including coordinate conventions, object rules, tool
dynamics, interaction memory, response schema, and feedback—is byte-identical.
The default released protocol uses `full`; `compact` documents the alternative
role framing supported by the runner. These files specify the prompt text,
not the availability of every model/variant combination.

## Initial action request

The runner inserts the goal, attempt budget, object-specific scene rule, tool
mechanism, and prediction targets into this exact template:

```text
Goal: {goal_text}

You may be asked for up to {attempt_budget} valid candidate placements. Each candidate is simulated independently from the same initial scene. The experimenter controls whether another candidate is requested and what evidence is supplied, so continuation alone does not reveal the preceding outcome.

{representation_note}

The environment uses a 600 x 600 coordinate space:
- Coordinates are integer pixels in the range 0-599.
- The origin (0,0) is at the bottom-left corner.
- The x coordinate increases from left to right.
- The y coordinate increases from bottom to top.
{scene_rules}
- You may drop one orange ball tool with diameter 72 pixels.
- {tool_dynamics}
- Choose a center point where the whole orange ball fits.
- Do not choose a placement that overlaps an existing object or scene boundary.

Interaction context:
- Your earlier action coordinates and coordinate predictions remain in the text conversation.
- Any success/failure labels revealed by this condition remain in the text conversation.
- Rollout media do not persist. On a continuation turn, only the latest rollout's 32 images or 32 JSON states are supplied.

{response_schema}
```

`scene_rules` identifies black objects as fixed, the red ball and blue objects
as movable, and the green container as fixed or movable according to the
world JSON. `tool_dynamics` explicitly states whether the orange tool
accelerates upward or downward, while all other movable objects accelerate
downward.

The response contains one action and a prospective terminal-state prediction
for every listed movable object, including the placed tool:

```json
{
  "action": "drop",
  "point": [300, 420],
  "coordinate_predictions": [
    {
      "id": "Ball",
      "state": "in_scene",
      "point": [250, 180],
      "orientation_deg": null
    }
  ],
  "reasoning": "brief reasoning"
}
```

For an exited object, `state` is `exited`, `point` is null, and `exit_side` is
one of `left`, `right`, `top`, or `bottom`. Orientation is requested only when
meaningful for a non-circular object.

## Feedback arms

The main comparisons use `full`, `neither`, and `trace_status`.
The additional canonical controls include `frames_only` and `status_only`.
Visual-initialized branches (`full`, `frames_only`, `status_only`, and
`neither`, where present) reuse the same attempt-1 model response and rollout.
The JSON representation control (`trace_status`) has an independent attempt 1
because it receives observable scene JSON instead of an image.

| Condition | Latest rollout | Explicit status |
|---|---|---|
| `full` | 32 uniformly sampled visual frames | success/failure |
| `trace_status` | 32 uniformly sampled observable JSON states | success/failure |
| `frames_only` | 32 uniformly sampled visual frames | withheld |
| `status_only` | withheld | success/failure |
| `neither` | withheld | withheld |

On every continuation, the initial scene is supplied again as either the
static image or observable scene JSON. Only the most recent 32-frame/32-state
rollout is supplied; earlier rollout media are not retained. Earlier actions,
predictions, and any status labels actually revealed by that condition remain
as text.

Status-visible arms stop at the first success. In the three main models,
an immediately successful shared first action is absorbed by all visual
branches; after a shared first failure, status-hidden branches continue to
eight valid attempts. Robotics-ER 2 status-hidden branches continue to eight
even after an immediately successful action. Invalid, obstructed and duplicate
proposals are repaired and do not count as valid attempts. Completion reasons
and coordinates are recorded in each released episode summary.

## Isolated held-out prediction branches

The protocol uses two fixed untried placements within each episode:

- The early probe is predicted before feedback and after exactly one rollout.
- The terminal probe is predicted before interaction and at the terminal checkpoint.

The same placement is used within each pre/post comparison; the early and
terminal placements may differ. Probe queries run in isolated conversation
branches and never supply prediction-error feedback to the action-selection
conversation. The matched terminal analysis requires the probe to remain at
least 40 pixels from executed actions; early-probe eligibility uses the first action.

Primary prediction scoring concerns terminal object states, exit sides and
spatial endpoints. Optional orientation and coarse-location diagnostics are
separate from primary PQ. Movement/collision events used for task goals are
not separately scored as prospective event predictions.

## Authoritative source

Dynamic wording and all format-only/geometric-validity repair prompts are
generated by:

- [`scripts/experiment/forked_feedback_runner.py`](../scripts/experiment/forked_feedback_runner.py)
- [`scripts/experiment/transfer_sidecar_runner.py`](../scripts/experiment/transfer_sidecar_runner.py)

Released episodes in [`data/finalized/`](../data/finalized/) contain the
rendered user prompts, prompt SHA-256, supplied image hashes, raw responses,
parsed predictions and provider metadata. The episode manifest identifies the
model and prompt variant. The exact system strings are
[`system_full.txt`](system_full.txt) and [`system_compact.txt`](system_compact.txt).

A concrete two-attempt full-feedback cycle, including the rendered prompts,
prospective coordinate predictions, failure feedback, next action, and
stopping rule, is reproduced in
[`docs/worked_interaction_example.md`](../docs/worked_interaction_example.md).
