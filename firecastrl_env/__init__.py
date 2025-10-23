from gymnasium.envs.registration import register
import envs.config as config

register(
    id="firecastrl/Wildfire-env0",
    entry_point="firecastrl_env.envs:WildfireEnv",
    max_episode_steps=config.MAX_TIMESTEPS
)
