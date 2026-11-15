# Worked full-feedback interaction

This is one completed, schema-valid GPT-5 unit from seed 2026. It is included
to make the context and feedback protocol concrete. The goal was:

> blue plank ends touching orange ball tool and partially inside the green
> container

The orange tool and all other movable objects accelerated downward. The
attempt budget was eight.

## Attempt 1 request

The model received the static initial scene and this rendered user prompt:

```text
Goal: blue plank ends touching orange ball tool and partially inside the green container

You may be asked for up to 8 valid candidate placements. Each candidate is simulated independently from the same initial scene. The experimenter controls whether another candidate is requested and what evidence is supplied, so continuation alone does not reveal the preceding outcome.

The first attached image is the static initial scene.

The environment uses a 600 x 600 coordinate space:
- Coordinates are integer pixels in the range 0-599.
- The origin (0,0) is at the bottom-left corner.
- The x coordinate increases from left to right.
- The y coordinate increases from bottom to top.
- Black objects are fixed and cannot be moved.
- The red ball and all blue objects are movable.
- The green goal container is movable.
- You may drop one orange ball tool with diameter 72 pixels.
- After placement, the orange tool is released and accelerates downward, as do the other movable objects.
- Choose a center point where the whole orange ball fits.
- Do not choose a placement that overlaps an existing object or scene boundary.

Interaction context:
- Your earlier action coordinates and coordinate predictions remain in the text conversation.
- Any success/failure labels revealed by this condition remain in the text conversation.
- Rollout media do not persist. On a continuation turn, only the latest rollout's 32 images or 32 JSON states are supplied.

Before seeing the rollout, predict the terminal state of every listed movable object:
- id "Ball": red ball (Ball)
- id "Goal": green container (Container)
- id "Support": blue plank (Poly)
- id "PLACED": orange ball tool (Ball)

For an object remaining in the scene, use {"id":"...","state":"in_scene","point":[x,y],"orientation_deg":number_or_null}.
For an object that exits, use {"id":"...","state":"exited","exit_side":"left|right|top|bottom","point":null,"orientation_deg":null}.
Use orientation_deg only when meaningful for a non-circular object; otherwise use null.

Return exactly one JSON object:
{"action":"drop","point":[x,y],"coordinate_predictions":[...one entry for every id above...],"reasoning":"<brief reasoning>"}
```

## Attempt 1 response and simulation

The model proposed `[250, 520]` and made prospective predictions before
seeing the rollout:

```json
{
  "action": "drop",
  "point": [250, 520],
  "coordinate_predictions": [
    {"id": "Ball", "state": "in_scene", "point": [300, 200], "orientation_deg": null},
    {"id": "Goal", "state": "in_scene", "point": [110, 70], "orientation_deg": 0},
    {"id": "Support", "state": "in_scene", "point": [170, 180], "orientation_deg": -30},
    {"id": "PLACED", "state": "in_scene", "point": [140, 120], "orientation_deg": null}
  ],
  "reasoning": "Dropping the large orange ball onto the mid/right of the blue plank should knock the red ball away and force the plank to tip off its support, with the left side dropping into the green container. The plank is likely to settle against the orange ball while partly inside the green container."
}
```

The rollout moved the red ball rightward, but the blue plank remained above
the green container rather than ending partially inside it. The simulator
therefore judged the attempt a failure.

The model's predicted and simulator-observed terminal centers were:

| Object | Predicted center | Observed center | Center error |
| --- | ---: | ---: | ---: |
| red ball (`Ball`) | `[300, 200]` | `[375.0, 215.0]` | 76.5 px |
| green container (`Goal`) | `[110, 70]` | `[100.0, 53.4]` | 19.4 px |
| blue plank (`Support`) | `[170, 180]` | `[217.5, 235.3]` | 72.9 px |
| orange tool (`PLACED`) | `[140, 120]` | `[328.7, 235.8]` | 221.5 px |

Under the paper's metrics, this placement had action quality **0.879** and
role-averaged prospective prediction quality **0.774**. Action quality is
`max(0, 1 - d/D_g)`, where `d` is distance to the closest successful
placement and `D_g` is the goal-specific normalizer over valid placements.
Here, `d = 54.1 px`. Prediction quality uses the paper-defined Gaussian
endpoint score with the 150-pixel primary scale, averaged across object roles.

## Attempt 2 context and request

The full-feedback branch retained the attempt-1 action and predictions in
text. It attached the unchanged static scene followed by exactly 32 uniformly
sampled images from the attempt-1 rollout. It then supplied:

```text
Latest recorded candidate:
- Attempt 1 was recorded at placement [250, 520].
- Goal: blue plank ends touching orange ball tool and partially inside the green container
- The first attached image is the unchanged static initial scene.
- The next 32 attached images are uniformly sampled frames from only the latest rollout, ordered early to late.
- Simulator result: the previous placement did not satisfy the goal.

You have 7 valid candidate placements remaining. Continuation does not by itself reveal the previous result. Use the information available in the conversation to submit a distinct next candidate.

Before seeing the rollout, predict the terminal state of every listed movable object:
- id "Ball": red ball (Ball)
- id "Goal": green container (Container)
- id "Support": blue plank (Poly)
- id "PLACED": orange ball tool (Ball)

For an object remaining in the scene, use {"id":"...","state":"in_scene","point":[x,y],"orientation_deg":number_or_null}.
For an object that exits, use {"id":"...","state":"exited","exit_side":"left|right|top|bottom","point":null,"orientation_deg":null}.
Use orientation_deg only when meaningful for a non-circular object; otherwise use null.

Return exactly one JSON object:
{"action":"drop","point":[x,y],"coordinate_predictions":[...one entry for every id above...],"reasoning":"<brief reasoning>"}
```

## Attempt 2 response and stopping rule

```json
{
  "action": "drop",
  "point": [160, 520],
  "coordinate_predictions": [
    {"id": "Ball", "state": "in_scene", "point": [360, 260], "orientation_deg": null},
    {"id": "Goal", "state": "in_scene", "point": [110, 70], "orientation_deg": 0},
    {"id": "Support", "state": "in_scene", "point": [150, 200], "orientation_deg": -25},
    {"id": "PLACED", "state": "in_scene", "point": [120, 120], "orientation_deg": null}
  ],
  "reasoning": "Dropping the orange tool onto the left side of the blue plank should drive that end down so the plank tips left and drops partially into the green container, settling while pressing against the orange ball. The red ball likely rolls off right toward the black container area."
}
```

This rollout drove the blue plank left and downward into the green container,
where it ended touching the orange tool. The red ball moved right and then
fell to the floor. The simulator-observed terminal centers were:

| Object | Predicted center | Observed center | Center error |
| --- | ---: | ---: | ---: |
| red ball (`Ball`) | `[360, 260]` | `[437.2, 16.0]` | 255.9 px |
| green container (`Goal`) | `[110, 70]` | `[100.5, 53.4]` | 19.1 px |
| blue plank (`Support`) | `[150, 200]` | `[109.5, 62.3]` | 143.5 px |
| orange tool (`PLACED`) | `[120, 120]` | `[52.6, 46.8]` | 99.5 px |

The simulator judged attempt 2 a success. Its action quality increased to
**0.997** (`d = 1.4 px`), while its prospective prediction quality decreased
to **0.665**. This concrete cycle therefore illustrates the distinction the
benchmark is designed to measure: feedback produced a substantially better
action without producing a more accurate prediction of that action's
consequences.

Because this status-visible branch succeeded, it stopped. In a status-hidden
branch, collection would continue to the fixed budget even if an earlier
attempt happened to satisfy the goal.

The corresponding JSON-feedback branch follows the same memory rule but
replaces the latest 32 images with 32 observable JSON states. Held-out
predictive-transfer queries use a fixed, unexecuted placement in an isolated
side branch and never rejoin this solver conversation.
