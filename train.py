import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import torch  # Added for neural network configuration
import time
from drone_env import DroneRLEnv


def test_environment():
    """Test if environment works correctly"""
    try:
        print("Testing Environment Setup...")
        env = DroneRLEnv()

        # Test reset
        obs, _ = env.reset()
        print(f"Reset successful. Observation shape: {obs.shape}")

        # Test step
        action = np.array([0.5, 0.5, 0.5, 0.5])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step successful. Reward: {reward}")

        env.close()
        return True

    except Exception as e:
        print(f"Environment test failed: {e}")
        return False


def train():
    # Create log directory with timestamp for multiple runs
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f"./tensorboard_logs/drone_training_{current_time}"
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([lambda: DroneRLEnv()])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device='cuda',
        tensorboard_log=log_dir,  # Added tensorboard logging
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],
                vf=[64, 64]
            ),
            activation_fn=torch.nn.ReLU
        )
    )

    try:
        print("Starting training...")
        model.learn(
            total_timesteps=2500000,
            tb_log_name="PPO",  # Name for this training run
            progress_bar=True  # Show progress bar
        )
        model.save(f"{log_dir}/drone_final")
        print("Training completed!")
        print(f"Logs saved to: {log_dir}")

    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        env.close()


def test_trained_model(model_path="drone_final"):
    """Test the trained model"""
    try:
        # Load model
        model = PPO.load(model_path)
        env = DroneRLEnv()

        # Run test episodes
        n_episodes = 5
        for episode in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0

            while True:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

    except Exception as e:
        print(f"Testing failed: {e}")

    finally:
        env.close()


def visualize_policy(model_path="drone_final", render_time=2):
    """Visualize the trained policy"""
    model = PPO.load(model_path)
    env = DroneRLEnv()

    print("\nVisualizing trained policy... Press Ctrl+C to stop")

    try:
        for episode in range(3):  # Show 3 episodes
            obs, _ = env.reset()
            episode_reward = 0.0  # Initialize as float

            print(f"\nEpisode {episode + 1}")
            print("Target height:", env.target_height)

            done = False
            while not done:

                # Get action from model
                action, _ = model.predict(obs)

                # Print current height and action
                #print(f"Height: {obs[2]:.2f}m | Actions: {[f'{a:.2f}' for a in action]}")

                # Execute action
                obs, reward, terminated, truncated, _ = env.step(action)
                if reward is not None:  # Add this check
                    episode_reward += reward
                done = terminated or truncated

                time.sleep(1 / 10)  # Slow down visualization

            print(f"Episode reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    # Create models directory
    os.makedirs("./models", exist_ok=True)

    # Test environment first
    print("Step 1: Testing environment...")
    if test_environment():
        print("Environment test successful!")

        # Train model
        print("\nStep 2: Starting training...")
        try:
            train()
        except Exception as e:
            print(f"Main execution failed: {e}")

        # Test trained model
        print("\nStep 3: Testing trained model...")
        test_trained_model()
    else:
        print("Environment test failed. Please fix issues before training.")

    print("\nStep 4: Visualizing trained policy...")
    visualize_policy()