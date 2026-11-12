def run_static_profile_baseline(backend, scenario: dict):
    # Uses complete background profile without selective querying
    p_known = scenario["full_profile"]
    sim = backend.simulate_choice(scenario["context"], p_known)
    is_match = 1.0 if (scenario["gold_choice"].lower() in sim.lower()) else 0.0
    # Queries = 5 because all 5 dimensions had to be pre-collected
    return {"choice_consistency": is_match, "query_count": 5, "words": len(sim.split())}
