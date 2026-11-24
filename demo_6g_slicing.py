"""
Demo: 6G Network Slicing with Reinforcement Learning
Demonstrates resource allocation using Gymnasium + Stable-Baselines3
"""

from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# ============================================================================
# Custom Environment: 6G Slicing Resource Allocation
# ============================================================================

class SixGSlicingEnv(gym.Env):
    """
    Simple 6G slicing environment:
    - 3 network slices: eMBB, URLLC, mMTC
    - Resource pool: 100 resource blocks
    - Action: allocate resources to each slice
    - Reward: maximize slice satisfaction
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Constants
        self.total_resources = 100  # Resource blocks
        self.num_slices = 3  # eMBB, URLLC, mMTC
        self.max_episodes_steps = 100
        self.current_step = 0

        # Action space: allocate resources to 3 slices (0-100 each, sum must be ≤100)
        # Simplified: each action is fraction of resources for each slice
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_slices,), dtype=np.float32
        )

        # Observation space: current demands and allocations
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.num_slices * 2,), dtype=np.float32
        )

        # Slice demands (traffic patterns)
        self.demands = np.array([
            30,  # eMBB: enhanced Mobile Broadband (high data rate)
            20,  # URLLC: Ultra-Reliable Low Latency Communication (low latency)
            15   # mMTC: massive Machine Type Communication (many devices)
        ], dtype=np.float32)

        self.allocations = np.zeros(self.num_slices, dtype=np.float32)
        self.episode_rewards = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.allocations = np.zeros(self.num_slices, dtype=np.float32)

        # Add some randomness to demands
        self.demands = np.array([
            self.np_random.integers(20, 40),  # eMBB
            self.np_random.integers(10, 25),  # URLLC
            self.np_random.integers(10, 20)   # mMTC
        ], dtype=np.float32)

        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        """Return current observation: [demands, allocations]"""
        return np.concatenate([self.demands, self.allocations]).astype(np.float32)

    def step(self, action):
        """
        Execute one step of environment.
        Action: [fraction for slice0, fraction for slice1, fraction for slice2]
        """
        self.current_step += 1

        # Normalize action to sum to 1, then scale to total resources
        action_normalized = action / (np.sum(action) + 1e-6)
        self.allocations = action_normalized * self.total_resources

        # Calculate satisfaction: min(allocation/demand, 1) for each slice
        satisfaction = np.minimum(self.allocations / (self.demands + 1e-6), 1.0)

        # Reward: sum of satisfactions (range 0-3)
        reward = float(np.sum(satisfaction))

        # Penalty for unfairness (variance in satisfaction)
        fairness_penalty = np.std(satisfaction) * 0.1
        reward -= fairness_penalty

        # Penalty for overallocation
        if np.sum(self.allocations) > self.total_resources:
            reward -= 10.0

        # Episode termination
        terminated = self.current_step >= self.max_episodes_steps
        truncated = False

        observation = self._get_obs()
        info = {
            "satisfaction": satisfaction.tolist(),
            "allocations": self.allocations.tolist(),
            "demands": self.demands.tolist(),
        }

        self.episode_rewards.append(reward)

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step {self.current_step}")
            print(f"  Demands:     {self.demands}")
            print(f"  Allocations: {self.allocations}")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    print("=" * 70)
    print("6G Network Slicing RL Demo")
    print("=" * 70)
    print()

    # Create environment
    print("Creating 6G slicing environment...")
    env = DummyVecEnv([lambda: SixGSlicingEnv()])

    # Train PPO agent
    print("Training PPO agent for 10,000 steps...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )
    model.learn(total_timesteps=10000)

    print("\n✓ Training complete!")
    print()

    # Test the trained model
    print("Testing trained agent...")
    test_env = SixGSlicingEnv()
    obs, _ = test_env.reset()

    test_rewards = []
    test_allocations = []
    test_satisfactions = []

    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        test_rewards.append(reward)
        test_allocations.append(info["allocations"])
        test_satisfactions.append(info["satisfaction"])

        if step % 25 == 0:
            print(f"  Step {step:3d}: Reward={reward:6.2f}, "
                  f"Satisfaction={np.mean(info['satisfaction']):5.2f}")

        if terminated or truncated:
            obs, _ = test_env.reset()

    print("\n✓ Testing complete!")
    print()

    # Results summary
    avg_reward = np.mean(test_rewards)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Test Satisfactions (mean): {np.mean(test_satisfactions, axis=0)}")
    print()

    # Visualization
    print("Generating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("6G Network Slicing RL Demo Results", fontsize=16, fontweight="bold")

    # Plot 1: Cumulative rewards
    ax = axes[0, 0]
    ax.plot(np.cumsum(test_rewards), linewidth=2, color="blue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Over Time")
    ax.grid(True, alpha=0.3)

    # Plot 2: Instantaneous rewards
    ax = axes[0, 1]
    ax.plot(test_rewards, alpha=0.7, linewidth=1, color="green")
    ax.axhline(y=np.mean(test_rewards), color="red", linestyle="--", label=f"Mean: {np.mean(test_rewards):.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Instantaneous Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Resource allocations over time
    ax = axes[1, 0]
    allocations_array = np.array(test_allocations)
    ax.plot(allocations_array[:, 0], label="eMBB (High Data)", linewidth=2)
    ax.plot(allocations_array[:, 1], label="URLLC (Low Latency)", linewidth=2)
    ax.plot(allocations_array[:, 2], label="mMTC (Many Devices)", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Allocated Resources")
    ax.set_title("Resource Allocation per Slice")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Slice satisfaction
    ax = axes[1, 1]
    satisfactions_array = np.array(test_satisfactions)
    ax.bar(
        ["eMBB", "URLLC", "mMTC"],
        np.mean(satisfactions_array, axis=0),
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_ylabel("Mean Satisfaction")
    ax.set_title("Average Slice Satisfaction")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / "6g_slicing_demo_results.png"
    plt.savefig(output_path, dpi=150)
    print(f"✓ Saved visualization to: 6g_slicing_demo_results.png")

    plt.show()

    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Integrate with ns-3 for network simulation (Phase 5)")
    print("  2. Add more complex slice requirements")
    print("  3. Implement multi-agent RL for distributed control")
    print()


if __name__ == "__main__":
    main()
