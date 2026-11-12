import os
import random
from tqdm import tqdm
from src.environment.ollama_client import OllamaBackend
from src.environment.mdp_env import AdeptSimEnv
from src.models.dqn_agent import DoubleDQNAgent
from src.data_loader import get_benchmark_scenarios

def main():
    print("="*60)
    print("Training AdeptSim RL Policy via Ollama")
    print("="*60)

    backend = OllamaBackend(llm_model="qwen2.5:7b", embed_model="nomic-embed-text")
    env = AdeptSimEnv(backend=backend, max_steps=5)
    
    state_dim = env.observation_space.shape[0]
    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=6, lr=1e-3)
    
    scenarios = get_benchmark_scenarios()
    num_episodes = 20
    epsilon = 0.4
    decay = 0.92

    for ep in range(num_episodes):
        scenario = random.choice(scenarios)
        state, _ = env.reset(scenario)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state, env.trait_mask, epsilon=epsilon)
            next_state, reward, done, _, info = env.step(action)
            
            agent.replay_buffer.append((state, action, reward, next_state, done))
            agent.update(batch_size=8)
            
            state = next_state
            total_reward += reward
            steps += 1

        epsilon = max(0.05, epsilon * decay)
        if (ep + 1) % 5 == 0:
            agent.sync_target()

        print(f"Episode {ep+1:02d}/{num_episodes} | Total Steps: {steps} | Cumulative Reward: {total_reward:+.2f} | Epsilon: {epsilon:.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    agent.save("checkpoints/adeptsim_dqn.pt")
    print("\nTraining completed! Model weights saved to checkpoints/adeptsim_dqn.pt")

if __name__ == "__main__":
    main()