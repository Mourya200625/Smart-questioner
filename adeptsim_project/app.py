
import streamlit as st
import numpy as np
import torch
import os

from src.environment.ollama_client import OllamaBackend
from src.models.dqn_agent import DoubleDQNAgent


# Page Configuration
st.set_page_config(
    page_title="AdeptSim | Adaptive Persona Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    .stApp {
        background-color: #0e1117;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .stButton button {
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Decision Domains
DOMAINS = [
    "Demographic & Role Constraints",
    "Core Values & Priorities",
    "Risk Tolerance & Stress Tendencies",
    "Social & Relational Dynamics",
    "Resource & Temporal Boundaries"
]


# Questions for Each Domain
DOMAIN_QUESTIONS = {
    "Demographic & Role Constraints":
        "What is your relevant background or role in this situation?",

    "Core Values & Priorities":
        "What core priority or personal value matters most to you here?",

    "Risk Tolerance & Stress Tendencies":
        "How do you feel about the risks or uncertainties in this choice?",

    "Social & Relational Dynamics":
        "What expectations do friends, family, or teammates have of you?",

    "Resource & Temporal Boundaries":
        "What constraints do you have regarding time, deadlines, or budget?"
}


# Domain Icons
DOMAIN_ICONS = {
    "Demographic & Role Constraints": "👤",
    "Core Values & Priorities": "💎",
    "Risk Tolerance & Stress Tendencies": "⚖️",
    "Social & Relational Dynamics": "🤝",
    "Resource & Temporal Boundaries": "⏳"
}


# Load Ollama Backend
@st.cache_resource
def load_backend():
    return OllamaBackend(
        llm_model="qwen2.5:7b",
        embed_model="nomic-embed-text"
    )


# Load DQN Agent
@st.cache_resource
def load_agent(state_dim=774):
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=6
    )

    ckpt_path = "checkpoints/adeptsim_dqn.pt"

    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)

    return agent


backend = load_backend()
agent = load_agent()


# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "p_known" not in st.session_state:
    st.session_state.p_known = {}

if "trait_mask" not in st.session_state:
    st.session_state.trait_mask = np.ones(5, dtype=np.float32)

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "is_finished" not in st.session_state:
    st.session_state.is_finished = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "scenario" not in st.session_state:
    st.session_state.scenario = ""

if "pending_domain" not in st.session_state:
    st.session_state.pending_domain = None


# Sidebar
with st.sidebar:

    st.title("🧠 AdeptSim Controller")

    st.caption("Budget-Aware Cold-Start Persona Completion")

    st.markdown("---")

    st.subheader("🎯 Trait Acquisition Status")

    for i, domain in enumerate(DOMAINS):

        icon = DOMAIN_ICONS[domain]

        is_queried = st.session_state.trait_mask[i] == 0.0

        st.markdown(f"**{icon} {domain}**")

        if is_queried:
            st.success(
                st.session_state.p_known.get(domain, "N/A")
            )
        else:
            st.caption("Status: Pending")

    st.markdown("---")

    budget_col1, budget_col2 = st.columns(2)

    budget_col1.metric(
        "Queries Used",
        f"{st.session_state.current_step}/5"
    )

    cost_penalty = st.session_state.current_step * -0.05

    budget_col2.metric(
        "API Cost Penalty",
        f"{cost_penalty:.2f}"
    )

    if st.button("🔄 Reset Conversation", use_container_width=True):

        st.session_state.chat_history = []
        st.session_state.p_known = {}
        st.session_state.trait_mask = np.ones(5, dtype=np.float32)
        st.session_state.current_step = 0
        st.session_state.is_finished = False
        st.session_state.prediction = None
        st.session_state.scenario = ""
        st.session_state.pending_domain = None

        st.rerun()


# Main Application
st.title("🧠 AdeptSim: Cold-Start Persona Simulator")

st.markdown(
    "An RL policy that dynamically identifies the "
    "**most critical trait domains** to query, "
    "minimizing unnecessary questioning before "
    "simulating your authentic decision."
)


# Step 1: Scenario Input
if not st.session_state.scenario:

    st.markdown("### 📝 Step 1: Present a Dilemma or Decision Scenario")

    with st.form("scenario_form"):

        scenario_input = st.text_area(
            "Enter the situation you need a simulated decision on:",
            placeholder="e.g., Should I quit my stable corporate job to join a high-risk early-stage AI startup?",
            height=100
        )

        submitted = st.form_submit_button(
            "Start Adaptive Elicitation 🚀",
            use_container_width=True
        )

    if submitted and scenario_input.strip():

        st.session_state.scenario = scenario_input.strip()

        context_emb = backend.get_embedding(
            st.session_state.scenario
        )

        st.session_state.context_emb = context_emb

        step_feat = np.array([0.0], dtype=np.float32)

        state = np.concatenate([
            context_emb,
            st.session_state.trait_mask,
            step_feat
        ]).astype(np.float32)

        action = agent.select_action(
            state,
            st.session_state.trait_mask,
            epsilon=0.0
        )

        if action == 5:
            st.session_state.is_finished = True

        else:
            st.session_state.pending_domain = DOMAINS[action]
            st.session_state.pending_action_idx = action

            st.session_state.chat_history.append({
                "role": "agent",
                "domain": DOMAINS[action],
                "text": DOMAIN_QUESTIONS[DOMAINS[action]]
            })

        st.rerun()


else:

    # Display Scenario
    st.info(f"**Current Scenario:** {st.session_state.scenario}")


    # Display Conversation History
    for chat in st.session_state.chat_history:

        if chat["role"] == "agent":

            st.markdown(
                f"**🤖 AdeptSim Agent ({chat.get('domain', 'Inquiry')}):**"
            )

            st.write(chat["text"])

        else:

            st.markdown("**You:**")

            st.write(chat["text"])


    # Step 2: User Response
    if not st.session_state.is_finished:

        with st.form("user_response_form", clear_on_submit=True):

            user_response = st.text_input(
                "Your Response:",
                placeholder="Type your answer here..."
            )

            send_btn = st.form_submit_button(
                "Send Response 💬",
                use_container_width=True
            )

        if send_btn and user_response.strip():

            # Save User Response
            current_domain = st.session_state.pending_domain

            st.session_state.p_known[current_domain] = user_response.strip()

            st.session_state.trait_mask[
                st.session_state.pending_action_idx
            ] = 0.0

            st.session_state.current_step += 1

            st.session_state.chat_history.append({
                "role": "user",
                "text": user_response.strip()
            })


            # Check Query Budget
            if st.session_state.current_step >= 5:

                st.session_state.is_finished = True

            else:

                step_feat = np.array([
                    st.session_state.current_step / 5.0
                ], dtype=np.float32)

                state = np.concatenate([
                    st.session_state.context_emb,
                    st.session_state.trait_mask,
                    step_feat
                ]).astype(np.float32)

                action = agent.select_action(
                    state,
                    st.session_state.trait_mask,
                    epsilon=0.0
                )

                # Terminal Stop Action
                if action == 5:

                    st.session_state.is_finished = True

                else:

                    st.session_state.pending_domain = DOMAINS[action]

                    st.session_state.pending_action_idx = action

                    st.session_state.chat_history.append({
                        "role": "agent",
                        "domain": DOMAINS[action],
                        "text": DOMAIN_QUESTIONS[DOMAINS[action]]
                    })

            st.rerun()


    # Step 3: Final Simulation
    if st.session_state.is_finished:

        st.success(
            "✅ **Stopping Criterion Met:** "
            "The RL policy acquired sufficient high-impact traits "
            "without redundant querying."
        )

        if not st.session_state.prediction:

            with st.spinner(
                "🔮 Downstream Simulator generating decision from acquired persona..."
            ):

                pred = backend.simulate_choice(
                    st.session_state.scenario,
                    st.session_state.p_known,
                    temperature=0.2
                )

                st.session_state.prediction = pred


        st.markdown("### 🎯 Simulated Decision Output")

        st.write(st.session_state.prediction)

        st.markdown("---")

        st.caption(
            f"Generated using conditioned traits in "
            f"{st.session_state.current_step} queries "
            f"(saved {5 - st.session_state.current_step} "
            f"potential domain requests)."
        )
