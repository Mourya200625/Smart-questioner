import numpy as np
from tabulate import tabulate
from src.environment.ollama_client import OllamaBackend
from src.environment.mdp_env import AdeptSimEnv
from src.models.dqn_agent import DoubleDQNAgent
from src.baselines.heuristic_stopping import run_heuristic_baseline
from src.baselines.fixed_budget import run_fixed_budget_baseline
from src.baselines.static_profile import run_static_profile_baseline
from src.data_loader import get_benchmark_scenarios

def evaluate():
    print("="*60)
    print("Evaluating AdeptSim vs Baselines on Benchmark Scenarios")
    print("="*60)

    backend = OllamaBackend(llm_model="qwen2.5:7b", embed_model="nomic-embed-text")
    env = AdeptSimEnv(backend=backend)
    scenarios = get_benchmark_scenarios()

    agent = DoubleDQNAgent(state_dim=env.observation_space.shape[0], action_dim=6)
    try:
        agent.load("checkpoints/adeptsim_dqn.pt")
        print("Loaded trained RL policy from checkpoints/adeptsim_dqn.pt\n")
    except Exception:
        print("No checkpoint found. Running evaluation with initialized policy.\n")

    results = {
        "AdeptSim (RL Policy)": {"matches": [], "queries": [], "words": []},
        "Heuristic Confidence": {"matches": [], "queries": [], "words": []},
        "Fixed Budget (k=2)": {"matches": [], "queries": [], "words": []},
        "Static Profile": {"matches": [], "queries": [], "words": []}
    }

    for sc in scenarios:
        # 1. RL Policy evaluation
        state, _ = env.reset(sc)
        done = False
        queries = 0
        while not done:
            action = agent.select_action(state, env.trait_mask, epsilon=0.0)
            state, reward, done, _, info = env.step(action)
            if not info["is_terminal"]:
                queries += 1
            else:
                sim = info.get("simulated_choice", "")
                results["AdeptSim (RL Policy)"]["matches"].append(info["choice_consistency"])
                results["AdeptSim (RL Policy)"]["queries"].append(queries)
                results["AdeptSim (RL Policy)"]["words"].append(len(sim.split()))

        # 2. Heuristic Confidence baseline
        res_heur = run_heuristic_baseline(backend, sc)
        results["Heuristic Confidence"]["matches"].append(res_heur["choice_consistency"])
        results["Heuristic Confidence"]["queries"].append(res_heur["query_count"])
        results["Heuristic Confidence"]["words"].append(res_heur["words"])

        # 3. Fixed Budget baseline
        res_fix = run_fixed_budget_baseline(backend, sc, budget=2)
        results["Fixed Budget (k=2)"]["matches"].append(res_fix["choice_consistency"])
        results["Fixed Budget (k=2)"]["queries"].append(res_fix["query_count"])
        results["Fixed Budget (k=2)"]["words"].append(res_fix["words"])

        # 4. Static Profile baseline
        res_stat = run_static_profile_baseline(backend, sc)
        results["Static Profile"]["matches"].append(res_stat["choice_consistency"])
        results["Static Profile"]["queries"].append(res_stat["query_count"])
        results["Static Profile"]["words"].append(res_stat["words"])

    # Format comparison table
    table_data = []
    for model_name, metrics in results.items():
        table_data.append([
            model_name,
            f"{np.mean(metrics['matches']) * 100:.1f}%",
            f"{np.mean(metrics['queries']):.2f}",
            f"{np.mean(metrics['words']):.1f}"
        ])

    headers = ["Framework / Baseline", "Choice Consistency (C_choice)", "Avg Queries (#requests)", "Avg Output Words"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    evaluate()