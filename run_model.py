import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from stable_baselines3 import PPO
from drone_env import DroneRLEnv
import time


def visualize_trained_model(model_path="drone_final", episodes=5):
    # Load model
    model = PPO.load(model_path)
    env = DroneRLEnv()

    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step = 0

            print(f"\nEpisode {episode + 1}")
            print("Target height: 1.0m")

            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

                if step % 20 == 0:  # Print every 20 steps
                    print(f"Height: {obs[2]:.2f}m | Reward: {reward:.2f}")

                step += 1
                time.sleep(1 / 30)  # Adjust for slower/faster visualization

            print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    visualize_trained_model()