
# interactive_chat.py
import torch
import numpy as np
import ollama

from src.environment.ollama_client import OllamaBackend
from src.models.dqn_agent import DoubleDQNAgent

DOMAINS = [
    "Demographic & Role Constraints",
    "Core Values & Priorities",
    "Risk Tolerance & Stress Tendencies",
    "Social & Relational Dynamics",
    "Resource & Temporal Boundaries"
]

# Conversational questions mapped to each of the 5 functional domains
DOMAIN_QUESTIONS = {
    "Demographic & Role Constraints": "What is your background or current role related to this situation?",
    "Core Values & Priorities": "What do you value most or prioritize in this scenario?",
    "Risk Tolerance & Stress Tendencies": "How do you feel about taking risks or dealing with potential downsides here?",
    "Social & Relational Dynamics": "What expectations do friends, family, or teammates have of you in this matter?",
    "Resource & Temporal Boundaries": "What constraints do you have regarding time, budget, or upcoming deadlines?"
}

def run_interactive_chatbot():
    backend = OllamaBackend(llm_model="qwen2.5:7b", embed_model="nomic-embed-text")
    
    # State dimension: 768 (embedding) + 5 (mask) + 1 (step progress)
    state_dim = 768 + 5 + 1
    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=6)
    
    try:
        agent.load("checkpoints/adeptsim_dqn.pt")
        print("\n[+] Loaded trained AdeptSim RL Policy checkpoint.")
    except Exception:
        print("\n[!] No checkpoint found. Running in interactive demo mode with base policy.")

    print("=" * 65)
    print("🤖 AdeptSim Interactive Persona Chatbot")
    print("=" * 65)
    print("The agent will ask you targeted questions until it has gathered")
    print("just enough traits to predict your decision without wasting your time.\n")

    scenario = input("Describe the scenario/decision you are facing:\n> ").strip()
    if not scenario:
        scenario = "Should I switch from a stable software job to joining an early-stage startup?"

    context_emb = backend.get_embedding(scenario)
    trait_mask = np.ones(5, dtype=np.float32)
    p_known = {}
    max_steps = 5
    current_step = 0

    while current_step <max_steps:
        # Build state
        step_feat = np.array([current_step / float(max_steps)], dtype=np.float32)
        state = np.concatenate([context_emb, trait_mask, step_feat]).astype(np.float32)

        # Policy selects the next best action (greedy, epsilon=0)
        action = agent.select_action(state, trait_mask, epsilon=0.0)

        # Action 5 = Stop Acquisition & Simulate
        if action == 5:
            print("\n🤖 Agent: 'I have gathered enough critical traits to understand your perspective. Stopping trait acquisition!'")
            break

        domain_name = DOMAINS[action]
        question = DOMAIN_QUESTIONS[domain_name]

        print(f"\n[Turn {current_step + 1}] Target Domain: {domain_name}")
        print(f"🤖 Agent: {question}")
        user_answer = input("You: ").strip()

        # Update known traits and mask out the queried domain
        p_known[domain_name] = user_answer
        trait_mask[action] = 0.0
        current_step += 1

    # Final Simulation
    print("\n" + "=" * 65)
    print("🔮 Simulating Your Decision with Acquired Persona...")
    print("=" * 65)
    
    predicted_decision = backend.simulate_choice(scenario, p_known, temperature=0.2)
    
    print(f"\nScenario: {scenario}")
    print("\nTraits Acquired by RL Policy:")
    for k, v in p_known.items():
        print(f" • {k}: {v}")
    
    print(f"\n🤖 Predicted Choice:\n{predicted_decision}\n")

if __name__ == "__main__":
    run_interactive_chatbot()
