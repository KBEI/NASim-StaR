"""Training script with action mask enabled.

This script trains a DQN agent on tiny-full-features scenario
with the enhanced action masking system enabled.
"""
import sys
import os
import importlib.util
import json
import time
from datetime import datetime

# Load local NASim
nasim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location("nasim", os.path.join(nasim_path, "__init__.py"))
nasim = importlib.util.module_from_spec(spec)
sys.modules["nasim"] = nasim
spec.loader.exec_module(nasim)

sys.path.insert(0, nasim_path)

from agents.dqn_agent import DQNAgent
import numpy as np


def train_with_mask(scenario_path, training_steps=50000, output_dir="runs/mask_test"):
    """Train agent with action masking"""
    
    print("=" * 80)
    print("Training DQN Agent with Action Masking")
    print("=" * 80)
    print(f"Scenario: {scenario_path}")
    print(f"Training steps: {training_steps:,}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment
    env = nasim.load(scenario_path, flat_actions=True, flat_obs=True)
    print(f"Environment loaded: {env.name}")
    print(f"  Action space: {env.action_space.n}")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Hosts: {len(env.scenario.hosts)}")
    print()
    
    # Create agent
    agent = DQNAgent(
        env,
        seed=42,
        lr=0.001,
        training_steps=training_steps,
        batch_size=32,
        replay_size=10000,
        final_epsilon=0.05,
        exploration_steps=int(training_steps * 0.5),
        gamma=0.99,
        hidden_sizes=[64, 64],
        target_update_freq=1000,
        verbose=True
    )
    
    # Training metrics
    episode_returns = []
    episode_steps = []
    episode_goals = []
    action_mask_stats = {
        'avg_valid_actions': [],
        'min_valid_actions': [],
        'max_valid_actions': []
    }
    
    print("\nStarting training...")
    start_time = time.time()
    
    # Train
    agent.train()
    
    # Collect episode statistics
    ep_count = 0
    ep_return = 0
    ep_step = 0
    ep_valid_actions = []
    
    obs, info = env.reset()
    
    for step in range(training_steps):
        # Get action mask
        mask = env.get_action_mask()
        ep_valid_actions.append(mask.sum())
        
        # Get epsilon
        epsilon = agent.get_epsilon()
        
        # Select action with mask
        action = agent.get_egreedy_action(obs, epsilon, action_mask=mask)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.replay.store(obs, action, next_obs, reward, done)
        
        # Update
        if agent.replay.size >= agent.batch_size and step % 4 == 0:
            agent.optimize()
        
        # Update episode stats
        ep_return += reward
        ep_step += 1
        obs = next_obs
        
        # Episode ended
        if done:
            episode_returns.append(ep_return)
            episode_steps.append(ep_step)
            episode_goals.append(int(env.goal_reached()))
            
            # Action mask stats
            if ep_valid_actions:
                action_mask_stats['avg_valid_actions'].append(np.mean(ep_valid_actions))
                action_mask_stats['min_valid_actions'].append(np.min(ep_valid_actions))
                action_mask_stats['max_valid_actions'].append(np.max(ep_valid_actions))
            
            ep_count += 1
            
            # Print progress
            if ep_count % 10 == 0:
                recent_returns = episode_returns[-10:]
                recent_goals = episode_goals[-10:]
                recent_steps = episode_steps[-10:]
                avg_valid = np.mean(action_mask_stats['avg_valid_actions'][-10:])
                
                print(f"Episode {ep_count:4d} | "
                      f"Steps: {step:6d}/{training_steps:6d} | "
                      f"Return: {np.mean(recent_returns):6.2f} | "
                      f"Steps/ep: {np.mean(recent_steps):5.1f} | "
                      f"Success: {sum(recent_goals)}/10 | "
                      f"Avg valid actions: {avg_valid:.1f}")
            
            # Reset for next episode
            ep_return = 0
            ep_step = 0
            ep_valid_actions = []
            obs, info = env.reset()
    
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Total episodes: {ep_count}")
    print(f"Final success rate (last 100): {sum(episode_goals[-100:])/min(100, len(episode_goals)):.1%}")
    
    # Save results
    results = {
        'scenario': scenario_path,
        'training_steps': training_steps,
        'total_episodes': ep_count,
        'elapsed_time': elapsed,
        'episode_returns': [float(r) for r in episode_returns],
        'episode_steps': [int(s) for s in episode_steps],
        'episode_goals': [int(g) for g in episode_goals],
        'action_mask_stats': {
            'avg_valid_actions': [float(x) for x in action_mask_stats['avg_valid_actions']],
            'min_valid_actions': [int(x) for x in action_mask_stats['min_valid_actions']],
            'max_valid_actions': [int(x) for x in action_mask_stats['max_valid_actions']]
        },
        'final_metrics': {
            'success_rate_last_100': sum(episode_goals[-100:])/min(100, len(episode_goals)),
            'avg_return_last_100': float(np.mean(episode_returns[-100:])),
            'avg_steps_last_100': float(np.mean(episode_steps[-100:]))
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    scenario_path = "/home/NASimLearn/NASim-StaR/scenarios/benchmark/tiny-full-features.yaml"
    training_steps = 100000  # Quick test
    
    results = train_with_mask(scenario_path, training_steps)
    
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Success rate (last 100): {results['final_metrics']['success_rate_last_100']:.1%}")
    print(f"Average return (last 100): {results['final_metrics']['avg_return_last_100']:.2f}")
    print(f"Average steps (last 100): {results['final_metrics']['avg_steps_last_100']:.1f}")
    
    if results['action_mask_stats']['avg_valid_actions']:
        avg_mask_size = np.mean(results['action_mask_stats']['avg_valid_actions'])
        print(f"Average valid actions per step: {avg_mask_size:.1f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
