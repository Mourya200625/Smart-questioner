from collections import Counter
import math

def calculate_semantic_entropy(backend, context: str, p_known: dict, n_rollouts: int = 3) -> float:
    """
    Samples multiple simulator decisions at temperature=0.7.
    A low entropy (< 0.4) indicates high consensus, meaning the agent has enough traits to halt.
    """
    rollouts = []
    for _ in range(n_rollouts):
        pred = backend.simulate_choice(context, p_known, temperature=0.7)
        # Normalize to the first 40 characters for key phrase clustering
        rollouts.append(pred.lower()[:40])

    counts = Counter(rollouts)
    total = len(rollouts)
    entropy = -sum((cnt / total) * math.log(cnt / total + 1e-9) for cnt in counts.values())
    return float(entropy)
