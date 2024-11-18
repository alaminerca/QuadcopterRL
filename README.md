# QuadcopterRL: Reinforcement Learning for Drone Stabilization

A PyBullet-based drone simulation using PPO (Proximal Policy Optimization) for learning hover and stabilization behaviors.

## Quick Start

```bash
# Create environment
conda create -n drone_rl python=3.9
conda activate drone_rl

# Install dependencies
pip install -r requirements.txt

# Run training (optional, if you want to train from scratch)
python train.py

# Run only visualization with pre-trained model
python run_model.py
```

## Run Pre-trained Model

Create a new file `run_model.py`:

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

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
                time.sleep(1/30)  # Adjust for slower/faster visualization
            
            print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        env.close()

if __name__ == "__main__":
    visualize_trained_model()
```

## Training Details

The model was trained for 2,500,000 iterations with the following parameters:
- Learning rate: 3e-4
- Batch size: 64
- Policy: MLP [64, 64]
- Value network: MLP [64, 64]
- Activation: ReLU
- GAE Lambda: 0.95
- Entropy coefficient: 0.01

## Environment Parameters

The drone environment uses realistic physics parameters:
- Drone mass: 0.5 kg (body) + 0.2 kg (rotors)
- Target height: 1.0m
- Max force per rotor: 2.58N
- Hover force needed: 6.87N
- Force per rotor at hover: 1.72N

## Project Structure
```
drone_rl_project/
├── drone_env.py          # Drone environment with PyBullet physics
├── train.py             # Training script with PPO
├── run_model.py         # Script to run trained model
├── requirements.txt     # Project dependencies
├── models/              # Saved model checkpoints
│   └── drone_final      # Trained for 2.5M steps
└── tensorboard_logs/    # Training metrics
```

## Visualization Controls
- Speed: Adjust `time.sleep(1/30)` in `run_model.py`
  - 1/60: Fast playback
  - 1/30: Normal speed
  - 1/20: Slow motion
  - 1/10: Very slow

## Training Metrics
View training progress:
```bash
tensorboard --logdir tensorboard_logs
```

## License
MIT License

## Contributing
Issues and pull requests are welcome!
