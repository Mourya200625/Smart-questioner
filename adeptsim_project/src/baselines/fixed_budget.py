DOMAINS = [
    "Demographic & Role Constraints",
    "Core Values & Priorities",
    "Risk Tolerance & Stress Tendencies",
    "Social & Relational Dynamics",
    "Resource & Temporal Boundaries"
]

def run_fixed_budget_baseline(backend, scenario: dict, budget: int = 2):
    p_known = {}
    for i in range(min(budget, len(DOMAINS))):
        domain = DOMAINS[i]
        val, _ = backend.query_user_proxy(scenario["full_profile"], domain)
        p_known[domain] = val

    sim = backend.simulate_choice(scenario["context"], p_known)
    is_match = 1.0 if (scenario["gold_choice"].lower() in sim.lower()) else 0.0
    return {"choice_consistency": is_match, "query_count": budget, "words": len(sim.split())}
