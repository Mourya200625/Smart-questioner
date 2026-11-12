import html
import os

import numpy as np
import streamlit as st
import torch  # noqa: F401  (kept: required by the DQN agent)

from src.environment.ollama_client import OllamaBackend
from src.models.dqn_agent import DoubleDQNAgent


# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AdeptSim | Adaptive Persona Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Theme: black ground, red-to-orange sun, glass cards, subtle motion
# ----------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
    --bg:#050505; --panel:#0B0B0B; --card:#101010; --line:#1E1E1E; --line2:#2A2A2A;
    --text:#FFFFFF; --muted:#A8A8A8; --accent:#FF5A1F; --accent2:#FFA23A; --deep:#C42A08;
}

/* ---------- base ---------- */
.stApp, .main { background: var(--bg) !important; color: var(--text); font-family:'Inter',sans-serif; }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { position: relative; z-index: 2; max-width: 820px; padding-top: 2rem; padding-bottom: 6rem; }
p, li, label, span, div { font-family:'Inter',sans-serif; }
.stApp p, .stApp li { color:#E6E6E6; }

/* ---------- background: faint grid, top glow, low rising sun ---------- */
.stApp {
    background:
        radial-gradient(ellipse 60% 40% at 50% 0%, rgba(255,90,31,.10), transparent 70%),
        var(--bg) !important;
}
.bg-grid {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
    background-size: 48px 48px;
    -webkit-mask-image: radial-gradient(ellipse 70% 60% at 50% 35%, #000 0%, transparent 75%);
            mask-image: radial-gradient(ellipse 70% 60% at 50% 35%, #000 0%, transparent 75%);
}
.orb-wrap { position: fixed; left: 0; right: 0; bottom: 0; height: 240px; overflow: hidden; z-index: 0; pointer-events: none; }
/* centre the sun on the main column when the sidebar is open */
.stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) .orb-wrap { left: 21rem; }
.orb {
    position: absolute; left: calc(50% - 400px); bottom: -600px; width: 800px; height: 800px; border-radius: 50%;
    background: radial-gradient(circle at 68% 38%, #FFA23A 0%, #F2490F 34%, #8C1006 68%, #220201 100%);
    box-shadow: 0 0 140px 30px rgba(240,70,15,.30);
    animation: float 7s ease-in-out infinite;
}
.orb-fade { position: absolute; inset: auto 0 0 0; height: 90px; background: linear-gradient(to top, #050505 5%, rgba(5,5,5,0)); }

/* ---------- type ---------- */
.mono { font-family:'JetBrains Mono',monospace; letter-spacing:.14em; text-transform:uppercase; font-size:11px; color: var(--muted); }
.hero { font-family:'Anton',sans-serif !important; font-weight:400; font-size:58px; line-height:1.08; text-align:center;
        margin: 14px auto 14px; letter-spacing:.01em; color:#fff !important; text-wrap: balance; animation: fadeUp .8s both; }
.hero span { display:block; color: var(--accent) !important; font-family:'Anton',sans-serif !important; }
.sub { display:block; width:min(560px,100%); margin:0 auto 28px; text-align:center; color:#C4C4C4; font-size:15px; line-height:1.65; text-wrap: balance; }
.center { text-align:center; }

/* ---------- glass cards + chat ---------- */
.glass { background: rgba(16,16,16,.78); border:1px solid var(--line2); border-radius:16px; backdrop-filter: blur(8px); }
.scenario { padding:14px 18px; display:flex; gap:14px; align-items:center; margin: 6px 0 18px; animation: fadeUp .8s both; }
.scenario .tag { color: var(--accent); flex:none; }
.msg { padding:16px 18px; max-width:84%; font-size:15px; line-height:1.55; margin: 0 0 14px; animation: fadeUp .6s both; color:#fff; }
.msg.agent { margin-right:auto; }
.msg.user { margin-left:auto; background: linear-gradient(135deg, var(--accent), var(--deep)); border-radius:16px; }
.msg .who { color: var(--accent); margin-bottom: 6px; display:block; }
.cursor { display:inline-block; width:8px; height:16px; background: var(--accent); vertical-align:-2px; margin-left:3px; animation: blink 1s steps(1) infinite; }

/* ---------- inputs & buttons ---------- */
div[data-testid="stForm"] { background: rgba(16,16,16,.78); border:1px solid var(--line2) !important; border-radius:18px; padding:16px; backdrop-filter: blur(8px); }
.stTextArea textarea, .stTextInput input {
    background:#0B0B0B !important; color:#fff !important; border:1px solid var(--line2) !important; border-radius:12px !important;
    font-family:'Inter',sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(255,90,31,.25) !important; }
.stTextArea label, .stTextInput label { color: var(--muted) !important; font-family:'JetBrains Mono',monospace !important; letter-spacing:.08em; font-size:12px !important; }

.stButton button, div[data-testid="stFormSubmitButton"] button {
    min-height: 46px; border-radius: 12px; font-weight:600; color:#fff; border:1px solid #333; background: transparent;
    transition: transform .2s, box-shadow .2s, border-color .2s;
}
.stButton button:hover { border-color: var(--accent); color:#fff; transform: translateY(-2px); }
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, var(--accent), var(--deep)); border: 0;
}
div[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-2px); box-shadow: 0 0 26px rgba(255,90,31,.55); color:#fff; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: .25rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .5rem; }
.brand { display:flex; align-items:center; gap:10px; font-family:'Anton',sans-serif; font-size:26px; letter-spacing:.04em; color:#fff; }
.trait { display:flex; gap:12px; align-items:flex-start; padding:10px 12px; margin-bottom:0; border-radius:12px;
         background: var(--card); border:1px solid var(--line); animation: fadeUp .7s both; }
.trait .idx { font-family:'JetBrains Mono',monospace; font-size:11px; color: var(--muted); padding-top:2px; }
.trait .name { font-size:13px; font-weight:600; color:#fff; }
.trait .state { font-size:12px; color: var(--muted); margin-top:4px; display:flex; align-items:center; gap:8px; }
.trait.done .state { color: var(--accent2); }
.trait.active { border-color: var(--accent); background: linear-gradient(135deg,#1c0a04,#101010); box-shadow: 0 0 28px rgba(255,90,31,.25); }
.trait.active .idx { color: var(--accent); }
.trait.active .state { color:#fff; }
.dot { width:8px; height:8px; border-radius:50%; background: var(--accent); display:inline-block; animation: pulse 1.4s infinite; }
.stat-row { display:flex; gap:10px; margin: 6px 0 12px; }
.stat { flex:1; padding:14px; }
.stat .num { font-family:'Anton',sans-serif; font-size:30px; margin-top:6px; color:#fff; }
.stat .num small { color: var(--muted); font-size:18px; }
.stat .num.warn { color: var(--accent2); }
.bar { height:6px; border-radius:3px; background: var(--line); overflow:hidden; margin-bottom: 18px; }
.bar i { display:block; height:100%; background: linear-gradient(90deg, var(--deep), var(--accent2), var(--deep)); background-size:200% 100%; animation: sweep 2.4s linear infinite; transition: width .6s ease; }

/* ---------- result ---------- */
.result-label { color: var(--accent); margin: 18px 0 8px; }
.done-banner { padding:14px 18px; margin: 12px 0; color:#fff; animation: fadeUp .6s both; border-color: rgba(255,162,58,.5); }

/* ---------- keyframes ---------- */
@keyframes fadeUp { from { opacity:0; transform: translateY(16px);} to { opacity:1; transform:none; } }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(255,90,31,.7);} 50% { box-shadow: 0 0 0 7px rgba(255,90,31,0);} }
@keyframes float { 0%,100% { transform: translateY(0);} 50% { transform: translateY(-22px);} }
@keyframes blink { 50% { opacity: 0; } }
@keyframes sweep { to { background-position: -200% 0; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
</style>

<div class="bg-grid"></div>
<div class="orb-wrap"><div class="orb"></div><div class="orb-fade"></div></div>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


def render(markup: str) -> None:
    """Render raw HTML. Lines are stripped so Markdown never treats indentation as a code block."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Decision domains
# ----------------------------------------------------------------------------
DOMAINS = [
    "Demographic & Role Constraints",
    "Core Values & Priorities",
    "Risk Tolerance & Stress Tendencies",
    "Social & Relational Dynamics",
    "Resource & Temporal Boundaries",
]

DOMAIN_QUESTIONS = {
    "Demographic & Role Constraints": "What is your relevant background or role in this situation?",
    "Core Values & Priorities": "What core priority or personal value matters most to you here?",
    "Risk Tolerance & Stress Tendencies": "How do you feel about the risks or uncertainties in this choice?",
    "Social & Relational Dynamics": "What expectations do friends, family, or teammates have of you?",
    "Resource & Temporal Boundaries": "What constraints do you have regarding time, deadlines, or budget?",
}

SHORT_NAMES = {
    "Demographic & Role Constraints": "Demographic & Role",
    "Core Values & Priorities": "Core Values & Priorities",
    "Risk Tolerance & Stress Tendencies": "Risk Tolerance & Stress",
    "Social & Relational Dynamics": "Social & Relational",
    "Resource & Temporal Boundaries": "Resource & Temporal",
}

STOP_ACTION = 5
MAX_QUERIES = 5


# ----------------------------------------------------------------------------
# Backend + agent
# ----------------------------------------------------------------------------
@st.cache_resource
def load_backend():
    return OllamaBackend(llm_model="qwen2.5:7b", embed_model="nomic-embed-text")


@st.cache_resource
def load_agent(state_dim=774):
    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=6)
    ckpt_path = "checkpoints/adeptsim_dqn.pt"
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
    return agent


backend = load_backend()
agent = load_agent()


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "chat_history": [],
        "p_known": {},
        "trait_mask": np.ones(5, dtype=np.float32),
        "current_step": 0,
        "is_finished": False,
        "prediction": None,
        "scenario": "",
        "pending_domain": None,
        "pending_action_idx": None,
        "context_emb": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state() -> None:
    st.session_state.chat_history = []
    st.session_state.p_known = {}
    st.session_state.trait_mask = np.ones(5, dtype=np.float32)
    st.session_state.current_step = 0
    st.session_state.is_finished = False
    st.session_state.prediction = None
    st.session_state.scenario = ""
    st.session_state.pending_domain = None
    st.session_state.pending_action_idx = None
    st.session_state.context_emb = None


init_state()


# ----------------------------------------------------------------------------
# RL policy step (same logic as before, shared by both entry points)
# ----------------------------------------------------------------------------
def ask_next_question() -> None:
    step_feat = np.array([st.session_state.current_step / MAX_QUERIES], dtype=np.float32)

    state = np.concatenate(
        [st.session_state.context_emb, st.session_state.trait_mask, step_feat]
    ).astype(np.float32)

    action = agent.select_action(state, st.session_state.trait_mask, epsilon=0.0)

    if action == STOP_ACTION:
        st.session_state.is_finished = True
        return

    domain = DOMAINS[action]
    st.session_state.pending_domain = domain
    st.session_state.pending_action_idx = action
    st.session_state.chat_history.append(
        {"role": "agent", "domain": domain, "text": DOMAIN_QUESTIONS[domain]}
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    render(
        """
        <div class="brand">
        <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
        <rect x="2" y="2" width="9" height="22" stroke="#FF5A1F" stroke-width="3"/>
        <path d="M15 2h9v22h-9" stroke="#fff" stroke-width="3"/></svg>
        ADEPTSIM</div>
        <div class="mono" style="margin:14px 0 22px">Budget-aware cold-start persona completion</div>
        <div class="mono" style="color:#fff;margin-bottom:12px">Trait acquisition</div>
        """
    )

    for i, domain in enumerate(DOMAINS):
        answered = st.session_state.trait_mask[i] == 0.0
        is_active = (
            not st.session_state.is_finished
            and not answered
            and st.session_state.pending_domain == domain
        )

        if answered:
            css, status = "done", "Captured"
        elif is_active:
            css, status = "active", '<span class="dot"></span>Asking now'
        else:
            css, status = "", "Pending"

        render(
            f"""
            <div class="trait {css}" style="animation-delay:{0.08 * i:.2f}s">
            <span class="idx">{i + 1:02d}</span>
            <div><div class="name">{SHORT_NAMES[domain]}</div>
            <div class="state">{status}</div></div></div>
            """
        )

    used = st.session_state.current_step
    penalty = used * -0.05
    pct = int(used / MAX_QUERIES * 100)

    render(
        f"""
        <div class="stat-row">
        <div class="glass stat"><div class="mono">Queries used</div>
        <div class="num">{used}<small>/{MAX_QUERIES}</small></div></div>
        <div class="glass stat"><div class="mono">Cost penalty</div>
        <div class="num warn">{penalty:.2f}</div></div></div>
        <div class="bar"><i style="width:{max(pct, 6)}%"></i></div>
        """
    )

    if st.button("Reset conversation", use_container_width=True):
        reset_state()
        st.rerun()


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
render(
    """
    <div class="mono center" style="margin-top:6px">RL policy · Adaptive elicitation · Persona agent</div>
    <div class="hero" role="heading" aria-level="1">Your Decision Is Already In Your Traits.<span>We Ask Only What Matters</span></div>
    <div class="sub">A reinforcement-learning policy picks the most critical trait domains to query,
    then simulates your authentic choice with the fewest questions possible.</div>
    """
)


# ----------------------------------------------------------------------------
# Step 1: scenario input
# ----------------------------------------------------------------------------
if not st.session_state.scenario:

    render('<div class="mono center" style="margin-bottom:12px">Step 1 · Present a dilemma or decision scenario</div>')

    with st.form("scenario_form"):
        scenario_input = st.text_area(
            "Enter the situation you need a simulated decision on:",
            placeholder="e.g., Should I quit my stable corporate job to join a high-risk early-stage AI startup?",
            height=100,
        )
        submitted = st.form_submit_button("Start adaptive elicitation", use_container_width=True)

    if submitted and scenario_input.strip():
        st.session_state.scenario = scenario_input.strip()
        st.session_state.context_emb = backend.get_embedding(st.session_state.scenario)
        ask_next_question()
        st.rerun()

else:
    # ------------------------------------------------------------------------
    # Scenario + conversation
    # ------------------------------------------------------------------------
    render(
        f"""
        <div class="glass scenario"><span class="mono tag">Scenario</span>
        <span style="font-size:14px;color:#fff">{html.escape(st.session_state.scenario)}</span></div>
        """
    )

    last_index = len(st.session_state.chat_history) - 1

    for idx, chat in enumerate(st.session_state.chat_history):
        text = html.escape(chat["text"])

        if chat["role"] == "agent":
            cursor = (
                '<span class="cursor"></span>'
                if idx == last_index and not st.session_state.is_finished
                else ""
            )
            render(
                f"""
                <div class="msg glass agent"><span class="mono who">Agent · {html.escape(chat.get("domain", "Inquiry"))}</span>
                {text}{cursor}</div>
                """
            )
        else:
            render(f'<div class="msg user">{text}</div>')

    # ------------------------------------------------------------------------
    # Step 2: user response
    # ------------------------------------------------------------------------
    if not st.session_state.is_finished:

        with st.form("user_response_form", clear_on_submit=True):
            user_response = st.text_input("Your response", placeholder="Type your answer here...")
            send_btn = st.form_submit_button("Send response", use_container_width=True)

        if send_btn and user_response.strip():
            current_domain = st.session_state.pending_domain
            st.session_state.p_known[current_domain] = user_response.strip()
            st.session_state.trait_mask[st.session_state.pending_action_idx] = 0.0
            st.session_state.current_step += 1
            st.session_state.chat_history.append({"role": "user", "text": user_response.strip()})

            if st.session_state.current_step >= MAX_QUERIES:
                st.session_state.is_finished = True
            else:
                ask_next_question()

            st.rerun()

    # ------------------------------------------------------------------------
    # Step 3: final simulation
    # ------------------------------------------------------------------------
    if st.session_state.is_finished:

        render(
            """
            <div class="glass done-banner"><span class="mono" style="color:#FFA23A">Stopping criterion met</span><br>
            The RL policy acquired sufficient high-impact traits without redundant querying.</div>
            """
        )

        if not st.session_state.prediction:
            with st.spinner("Downstream simulator generating a decision from the acquired persona..."):
                st.session_state.prediction = backend.simulate_choice(
                    st.session_state.scenario,
                    st.session_state.p_known,
                    temperature=0.2,
                )

        render('<div class="mono result-label">Simulated decision output</div>')
        st.write(st.session_state.prediction)

        saved = MAX_QUERIES - st.session_state.current_step
        render(
            f"""
            <div class="mono" style="margin-top:18px">
            Generated from conditioned traits in {st.session_state.current_step} queries
            (saved {saved} potential domain requests)</div>
            """
        )