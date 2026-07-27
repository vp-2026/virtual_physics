# Simulator

`runtime/` contains the Pygame/Pymunk virtual-tools simulator used to execute
tool placements. `goal_semantics/` contains rollout tracing, predicate
evaluation, and the simulator sweep used to construct goal banks and solution
densities.

Important frozen conventions:

- canvas: 600 × 600 pixels;
- origin: bottom-left;
- tool: 72-pixel-diameter orange ball;
- deterministic trace step: 0.02 seconds;
- maximum rollout: 30 seconds;
- canonical success: the red ball remains inside the green container
  continuously for at least two seconds;
- noncanonical success: the submitted endpoint/final-state evaluator saved
  with the paper goals.

Run `python simulator/goal_semantics/build_goal_bank_from_placement_sweep.py
--help` for the candidate sweep interface.
