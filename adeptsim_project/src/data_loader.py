def get_benchmark_scenarios():
    return [
        {
            "id": "scenario_01",
            "context": "Will Vikram drop out of the weekend hackathon to prepare for Monday's placement exam?",
            "gold_choice": "Stay for the hackathon",
            "full_profile": {
                "Demographic & Role Constraints": "20yo B.Tech student, designated hackathon backend architect.",
                "Core Values & Priorities": "Values practical project building over theoretical exam cramming.",
                "Risk Tolerance & Stress Tendencies": "High project risk tolerance, low exam anxiety.",
                "Social & Relational Dynamics": "Strong peer commitment to hackathon teammates.",
                "Resource & Temporal Boundaries": "Only has 36 hours before Monday morning.",
                "critical_domain": "Core Values & Priorities"
            }
        },
        {
            "id": "scenario_02",
            "context": "Will Ananya accept an immediate startup offer or wait for the Tier-1 consultancy result?",
            "gold_choice": "Wait for consultancy",
            "full_profile": {
                "Demographic & Role Constraints": "Final-year college student from Pune.",
                "Core Values & Priorities": "Values financial security and brand stability.",
                "Risk Tolerance & Stress Tendencies": "Extremely risk-averse regarding career stability.",
                "Social & Relational Dynamics": "Family expects a stable corporate trajectory.",
                "Resource & Temporal Boundaries": "Offer deadline expires in 48 hours.",
                "critical_domain": "Risk Tolerance & Stress Tendencies"
            }
        },
        {
            "id": "scenario_03",
            "context": "Will Rohan agree to lead the team presentation or delegate it to a teammate?",
            "gold_choice": "Delegate presentation",
            "full_profile": {
                "Demographic & Role Constraints": "Junior software developer.",
                "Core Values & Priorities": "Focuses on technical implementation and code quality.",
                "Risk Tolerance & Stress Tendencies": "High public speaking anxiety and introversion.",
                "Social & Relational Dynamics": "Collaborative relationship with senior teammates.",
                "Resource & Temporal Boundaries": "Presentation is scheduled for 3 PM tomorrow.",
                "critical_domain": "Risk Tolerance & Stress Tendencies"
            }
        }
    ]
