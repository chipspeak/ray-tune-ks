import random
from ray import tune
from ray.air import session

'''
The `config` dictionary is provided by Ray Tune for each trial run.
It contains values for hyperparameters — here, `x` and `y` — sampled from the defined search space.

The `objective` function simulates a training loop across 10 steps (iterations).
In each iteration, it:
- Extracts the current values of `x` and `y` from `config`
- Calculates a fake "loss" using a simple quadratic function
  (the function has its minimum when x = 3 and y = -1)
- Adds a small amount of random noise to simulate non-determinism
- Reports the current loss value back to Ray Tune using `session.report()`,
  which lets Tune track performance over time.
'''

def objective(config):
    for step in range(10):
        x, y = config["x"], config["y"]
        loss = (x - 3) ** 2 + (y + 1) ** 2 + random.random() * 0.1  # Simulated loss
        session.report({"loss": loss})  # Report metrics to Ray Tune


'''
We now tell Ray Tune to run the `objective` function as the training task.

- `config`: defines the search space for hyperparameters `x` and `y`.
  Both are sampled from a uniform distribution between -10 and 10,
  meaning any value in that range is equally likely to be picked.

- `num_samples`: runs 5 separate trials, each with different random values
  for `x` and `y`.

- `resources_per_trial`: reserves 1 CPU core for each trial (you could adjust this).
'''

tune.run(
    objective,
    config={
        "x": tune.uniform(-10, 10),
        "y": tune.uniform(-10, 10),
    },
    num_samples=5,
    resources_per_trial={"cpu": 1},
)
