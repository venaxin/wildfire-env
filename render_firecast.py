import gymnasium as gym
import numpy as np
import sys
import os
from stable_baselines3 import PPO

# --- IMPORTS FROM YOUR CUSTOM ENV ---
# Ensure the 'firecastrl_env' folder is in the same directory as this script
try:
    from firecastrl_env.envs.wildfire_env import WildfireEnv
except ImportError:
    print("Error: Could not find 'firecastrl_env'. Make sure the folder is in this directory.")
    sys.exit(1)

# --- WRAPPER DEFINITION ---
# We need the observation wrapper to ensure the model receives the same input format (no infs)
class SafeWildfireWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, obs):
        if 'cells' in obs:
            obs['cells'] = np.nan_to_num(obs['cells'], posinf=-1.0)
        return obs

def run_render(model_path, num_agents=3):
    print(f"Loading model from: {model_path}")
    
    # 1. Initialize Environment with render_mode='human'
    raw_env = WildfireEnv(num_agents=num_agents, render_mode="human")
    
    # 2. Wrap it (Crucial: Model expects observations processed by this wrapper)
    env = SafeWildfireWrapper(raw_env)
    
    # 3. Load the trained model
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return

    # 4. Run Loop
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    print("\n--- Starting Simulation (Press CTRL+C to stop) ---")
    try:
        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render happens automatically inside step() because of render_mode="human"
            # But we call it explicitly to be safe for some gym versions
            env.render()
            
            done = terminated or truncated
            
            if done:
                print(f"Episode Finished. Total Reward: {total_reward:.2f}")
                print(f"Cells Burnt: {info.get('cells_burnt', 'N/A')}")
                
                # Optional: Reset to keep watching
                user_input = input("Press Enter to run again, or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
                obs, _ = env.reset()
                done = False
                total_reward = 0
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Change this to the model you want to watch
    # Options based on your training:
    #   "ppo_fire_squad_coop.zip"
    #   "ppo_fire_squad_greedy.zip"
    #   "ppo_fire_squad_curriculum.zip"
    
    MODEL_FILENAME = "ppo_fire_squad_greedy.zip" 
    
    run_render(MODEL_FILENAME)