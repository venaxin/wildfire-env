## FirecastRL Wildfire Environment

Gymnasium-compatible wildfire simulation environment with ignition and spread dynamics. Packaged as `firecastrl-env`.

### Install

From source:
```bash
pip install -e .
```

After install, the environment is registered as `firecastrl/Wildfire-env0`.

### Quickstart

```python
import gymnasium as gym
import numpy as np

import firecastrl_env  # registers the env via entry points

env = gym.make("firecastrl/Wildfire-env0")
obs, info = env.reset()
done = False
truncated = False
total_reward = 0.0

while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

env.close()
print("Episode reward:", total_reward)
```

### Observation and Actions

- Observation: Dict with keys
  - `cells`: `(160, 240)` float32 array of ignition times (∞ for not scheduled to ignite)
  - `helicopter_coord`: `(2,)` int32 array, `[x, y]`
  - `quenched_cells`: `(1,)` float32 scalar for cells extinguished this step

- Action: Discrete(5)
  - 0: Down, 1: Up, 2: Left, 3: Right, 4: Helitack

### Seeding and Config

Config lives in `firecastrl_env/envs/config.py`. Key parameters:
`gridWidth`, `gridHeight`, `cellSize`, `MAX_TIMESTEPS`, `HELICOPTER_SPEED`.

### Development

Project layout:
```
firecastrl_env/
  __init__.py              # registers env id
  envs/
    __init__.py
    wildfire_env.py        # main env
    config.py
    environment/
      cell.py, enums.py, helper.py, vector.py, wind.py, zone.py
    fire_engine/
      fire_engine.py, fire_spread_rate.py, utils.py
```

Run a quick check:
```python
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import firecastrl_env

env = gym.make("firecastrl/Wildfire-env0")
check_env(env, warn=True, skip_render_check=True)
```

### License

MIT