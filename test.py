# test.py (Final Version)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from dino_env import DinoEnv
import time

# --- 1. CONFIGURE YOUR MODEL ---
# Find these values from your 'models' folder
SESSION_ID = "PPO_Upgraded-1760860107"  # The name of the folder for your training session
MODEL_TIMESTEP = 300000  # The number of steps for the model you want to test

MODEL_PATH = f"models/{SESSION_ID}/{MODEL_TIMESTEP}.zip"

# --- 2. INITIALIZE THE ENVIRONMENT WITH WRAPPERS ---
print("Loading model and initializing environment...")
vec_env = make_vec_env(DinoEnv, n_envs=1)
env = VecFrameStack(vec_env, n_stack=4)  # Must use the same wrapper as in training
model = PPO.load(MODEL_PATH, env=env)
print(f"Model {MODEL_PATH} loaded successfully!")

# --- 3. WATCH THE AI PLAY ---
for i in range(10):
    print(f"\n--- Starting Game #{i + 1} ---")

    obs = env.reset()
    done = False
    frames_survived = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        frames_survived += 1
        time.sleep(0.02)

    print(f"Game Over! Survived for {frames_survived} frames.")
    time.sleep(2)

# --- 4. CLOSE THE ENVIRONMENT ---
env.close()
print("\n--- Test complete! ---")