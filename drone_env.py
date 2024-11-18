import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class DroneRLEnv(gym.Env):
    """Custom Drone Environment that follows gym interface"""

    def __init__(self):
        super().__init__()

        # Initialize PyBullet with better camera
        try:
            p.disconnect()  # Disconnect any existing connections
        except:
            pass

        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # Set camera position for better view
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )

        # Set gravity and load ground
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load ground with different color
        self.plane = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

        # Create drone
        self.drone = self._create_drone()

        # Action space: [0,1] for each rotor
        self.action_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )

        # State space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # Parameters
        self.target_height = 1.0
        self.target_position = np.array([0, 0, self.target_height])
        self.max_steps = 1000
        self.current_step = 0

        high = np.array([
            10.0,  # x pos
            10.0,  # y pos
            10.0,  # z pos
            np.pi,  # roll
            np.pi,  # pitch
            np.pi,  # yaw
            10.0,  # x vel
            10.0,  # y vel
            10.0,  # z vel
            5.0,  # roll rate
            5.0,  # pitch rate
            5.0  # yaw rate
        ])

        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

    def get_state(self):
        """Gets the current state of the drone"""
        # Get position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.drone)

        # Convert quaternion to euler angles
        euler = p.getEulerFromQuaternion(orn)

        # Get linear and angular velocity
        lin_vel, ang_vel = p.getBaseVelocity(self.drone)

        # Combine into state vector
        state = np.array(pos + euler + lin_vel + ang_vel)

        return state

    def apply_action(self, action):
        # Keep physics calculations
        drone_mass = 0.5
        rotor_mass = 0.05 * 4
        total_mass = drone_mass + rotor_mass
        gravity = 9.81

        hover_force = total_mass * gravity
        force_per_rotor = hover_force / 4
        max_force = force_per_rotor * 1.5  # Reduced from 2.0 for more stability

        # Stronger smoothing
        if not hasattr(self, 'previous_forces'):
            self.previous_forces = np.zeros(4)

        # Increased smoothing factor
        smoothing = 0.8  # Increased from 0.7
        smoothed_action = smoothing * self.previous_forces + (1 - smoothing) * action
        self.previous_forces = smoothed_action

        # Add hover bias to help maintain altitude
        hover_bias = 0.5  # Base thrust to counteract gravity

        for i, force in enumerate(smoothed_action):
            # Combine hover bias with control action
            thrust = (hover_bias + force * 0.5) * max_force
            thrust = np.clip(thrust, 0, max_force)

            p.applyExternalForce(
                self.rotors[i],
                -1,
                [0, 0, thrust],
                [0, 0, 0],
                p.WORLD_FRAME
            )

            # Further reduced torque
            torque_magnitude = 0.0005 * thrust  # Reduced from 0.001
            torque_direction = 1 if i in [0, 2] else -1
            p.applyExternalTorque(
                self.rotors[i],
                -1,
                [0, 0, torque_magnitude * torque_direction],
                p.WORLD_FRAME
            )

        # Log more detailed information
        if self.current_step % 100 == 0:
            print(f"\nForce analysis:")
            print(f"Hover force needed: {hover_force:.2f}N")
            print(f"Force per rotor: {force_per_rotor:.2f}N")
            print(f"Max force: {max_force:.2f}N")
            print(f"Average thrust: {np.mean(smoothed_action * max_force):.2f}N")
            print(f"Action smoothness: {np.std(smoothed_action):.2f}")

    def compute_reward(self, state, action):
        position = state[0:3]
        orientation = state[3:6]
        velocities = state[6:9]
        ang_velocities = state[9:12]

        # Height control (more precise)
        height_diff = abs(position[2] - self.target_height)
        height_reward = 2.0 / (1.0 + height_diff * 5)  # Sharper dropoff

        # Bonus for very stable height
        height_bonus = 3.0 if height_diff < 0.05 else 0.0

        # Stronger penalties for instability
        tilt = abs(orientation[0]) + abs(orientation[1])
        tilt_penalty = -tilt * 3.0

        # Penalize rapid movements more
        velocity_penalty = -np.sum(np.square(velocities)) * 0.3
        ang_velocity_penalty = -np.sum(np.square(ang_velocities)) * 0.3

        # Encourage smooth, balanced control (penalize extreme actions)
        action_smoothness = -np.sum(np.square(action - 0.5)) * 0.5
        action_balance = -np.std(action) * 0.5  # Penalize uneven rotor usage

        reward = (
                height_reward +  # Base height control
                height_bonus +  # Precision bonus
                tilt_penalty +  # Stability
                velocity_penalty +  # Smooth motion
                ang_velocity_penalty +  # Rotation stability
                action_smoothness +  # Smooth control
                action_balance  # Balanced rotors
        )

        if self.current_step % 100 == 0:  # Reduce logging frequency
            print(f"\nStep {self.current_step}:")
            print(f"Height: {position[2]:.2f}m (target: {self.target_height}m)")
            print(f"Tilt: {tilt:.2f}")
            print(f"Action range: {np.min(action):.2f} to {np.max(action):.2f}")
            print(f"Reward: {reward:.2f}")

        return float(np.clip(reward, -10, 10))


    def is_terminated(self, state):
        """
        Check if episode should terminate:
        1. Crash detection (too low)
        2. Extreme tilt detection
        """
        position = state[0:3]
        orientation = state[3:6]

        # More lenient crash condition
        if position[2] < 0.02:  # Changed from 0.05
            return True

        # More lenient tilt condition
        if np.abs(orientation[0]) > np.pi / 2 or np.abs(orientation[1]) > np.pi / 2:  # Changed from pi/3
            return True

        return False

    def step(self, action):
        self.apply_action(action)  # Changed from _apply_action
        p.stepSimulation()

        state = self.get_state()
        reward = self.compute_reward(state, action)

        terminated = self.is_terminated(state)
        truncated = self.current_step >= self.max_steps

        self.current_step += 1

        return state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # Reset drone position and orientation
        p.resetBasePositionAndOrientation(self.drone, [0, 0, 0.5], [0, 0, 0, 1])

        # Reset rotor positions
        rotor_positions = [
            [0.1, 0.1, 0.52],
            [-0.1, 0.1, 0.52],
            [-0.1, -0.1, 0.52],
            [0.1, -0.1, 0.52]
        ]

        for motor, pos in zip(self.rotors, rotor_positions):
            p.resetBasePositionAndOrientation(motor, pos, [0, 0, 0, 1])

        state = self.get_state()  # Changed from _get_state to get_state
        return state, {}

    def _create_drone(self):
        # Main body
        body_mass = 0.5
        body_size = [0.06, 0.06, 0.02]
        self.body = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_size)

        # Create main body and store ID immediately
        self.drone = p.createMultiBody(
            baseMass=body_mass,
            baseCollisionShapeIndex=self.body,
            basePosition=[0, 0, 0.5],
            baseOrientation=[0, 0, 0, 1]
        )

        # Change body color to blue
        p.changeVisualShape(self.drone, -1, rgbaColor=[0, 0, 1, 0.8])

        # Create 4 rotors with smaller dimensions
        self.rotors = []
        rotor_mass = 0.05
        rotor_radius = 0.02
        rotor_height = 0.005

        # Rotor positions in X configuration
        rotor_positions = [
            [0.08, 0.08, 0.51],  # Front Right
            [-0.08, 0.08, 0.51],  # Front Left
            [-0.08, -0.08, 0.51],  # Rear Left
            [0.08, -0.08, 0.51]  # Rear Right
        ]

        # Different colors for each rotor
        rotor_colors = [
            [1, 0, 0, 0.8],  # Red - Front Right
            [0, 1, 0, 0.8],  # Green - Front Left
            [1, 1, 0, 0.8],  # Yellow - Rear Left
            [1, 0.5, 0, 0.8]  # Orange - Rear Right
        ]

        for pos, color in zip(rotor_positions, rotor_colors):
            rotor = p.createCollisionShape(p.GEOM_CYLINDER,
                                           radius=rotor_radius,
                                           height=rotor_height)
            motor = p.createMultiBody(
                baseMass=rotor_mass,
                baseCollisionShapeIndex=rotor,
                basePosition=pos
            )

            # Set rotor color
            p.changeVisualShape(motor, -1, rgbaColor=color)

            # Create constraint between body and rotor
            constraint = p.createConstraint(
                self.drone, -1, motor, -1,
                p.JOINT_FIXED, [0, 0, 0],
                parentFramePosition=[pos[0], pos[1], 0.01],  # Relative to body
                childFramePosition=[0, 0, 0],  # Center of rotor
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=[0, 0, 0, 1]
            )
            p.changeConstraint(constraint, maxForce=100)

            self.rotors.append(motor)

        return self.drone

    def close(self):
        """Properly closes PyBullet connection"""
        try:
            p.disconnect()
        except:
            pass
