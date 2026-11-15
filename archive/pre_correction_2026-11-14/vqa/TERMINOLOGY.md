# Terminology

## What “goal role” means

A **goal role** is an object’s functional relationship to the task, not merely
its visible color or shape.

- **Target:** the object whose state must change; in the canonical task, the
  red ball.
- **Goal or destination:** the desired receptacle or reference object; in the
  canonical task, the green container.
- **Intervention tool:** the orange object placed by the agent.
- **Other movable objects:** blue objects that can mediate the outcome.
- **Fixed structure:** black objects that constrain motion.

The static VQA uses the canonical visual semantics—red target and green
destination—to test whether the model can identify the objects relevant to the
task. Recognizing a goal role is counted as scene/task parsing, not as evidence
that the model understands the physics.

For noncanonical goals, roles are assigned from the goal statement rather than
hard-coded from color. The same physical object can therefore be a subject,
reference, support, obstacle, or irrelevant object in different goals.

## Coordinate convention

- Canvas: 600 × 600 pixels.
- Origin `(0,0)`: bottom-left.
- x increases left to right.
- y increases bottom to top.
- Valid integer coordinates: 0 through 599.
- A proposed action is the center of the 72-pixel-diameter orange tool.

## Layout versus cell

A **layout** is one initial visual world. A **cell** is a layout paired with one
tool mechanism. The release has 66 layouts and 132 cells.

