import ollama
import numpy as np

class OllamaBackend:
    def __init__(self, llm_model: str = "qwen2.5:7b", embed_model: str = "nomic-embed-text"):
        self.llm = llm_model
        self.embed_model = embed_model

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates dense vector embeddings for dialogue and scenario context."""
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)

    def query_user_proxy(self, full_profile: dict, domain_name: str) -> tuple[str, bool]:
        """
        User Proxy: Answers a query for an unrevealed trait domain.
        Returns the revealed attribute string and whether it resolves a critical dimension.
        """
        attribute_val = full_profile.get(domain_name, "No specific constraint recorded.")
        is_critical = (full_profile.get("critical_domain") == domain_name)
        return attribute_val, is_critical

    def simulate_choice(self, scenario_context: str, p_known: dict, temperature: float = 0.1) -> str:
        """
        Downstream Simulator: Predicts the individual's choice given the currently known attributes.
        """
        traits_summary = "\n".join([f"- {k}: {v}" for k, v in p_known.items()]) if p_known else "None"
        prompt = (
            f"You are simulating a persona in the following scenario:\n"
            f"Scenario: {scenario_context}\n\n"
            f"Known Traits:\n{traits_summary}\n\n"
            f"Question: What decision does this individual make?\n"
            f"Provide only the direct choice or one concise sentence stating the decision."
        )

        response = ollama.chat(
            model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response["message"]["content"].strip()