from gymnasium.envs.registration import register

register(
    id="firecastrl-env/GridWorld-v0",
    entry_point="firecastrl-env.envs:GridWorldEnv",
)
