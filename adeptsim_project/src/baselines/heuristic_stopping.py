import ollama

DOMAINS = [
    "Demographic & Role Constraints",
    "Core Values & Priorities",
    "Risk Tolerance & Stress Tendencies",
    "Social & Relational Dynamics",
    "Resource & Temporal Boundaries"
]

def run_heuristic_baseline(backend, scenario: dict, confidence_threshold: int = 75):
    p_known = {}
    queries = 0
    context = scenario["context"]

    for domain in DOMAINS:
        # Verbalized confidence prompt
        traits_str = "\n".join([f"- {k}: {v}" for k, v in p_known.items()]) if p_known else "None"
        prompt = (
            f"Scenario: {context}\n"
            f"Known Traits:\n{traits_str}\n"
            f"Rate your confidence from 0 to 100 on whether you have enough traits to accurately predict the decision.\n"
            f"Return ONLY the number."
        )
        try:
            res = ollama.chat(model=backend.llm, messages=[{"role": "user", "content": prompt}])
            conf = int("".join(filter(str.isdigit, res["message"]["content"][:10])))
        except Exception:
            conf = 50

        if conf >= confidence_threshold:
            break

        val, _ = backend.query_user_proxy(scenario["full_profile"], domain)
        p_known[domain] = val
        queries += 1

    sim = backend.simulate_choice(context, p_known)
    is_match = 1.0 if (scenario["gold_choice"].lower() in sim.lower()) else 0.0
    return {"choice_consistency": is_match, "query_count": queries, "words": len(sim.split())}
