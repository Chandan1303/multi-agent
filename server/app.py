

"""
FastAPI application for the Smart City Openenv Environment.

This module creates an HTTP server that exposes the SmartCityOpenenvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SmartCityOpenenvAction, SmartCityOpenenvObservation
    from .smart_city_openenv_environment import SmartCityOpenenvEnvironment
    from ..agents.industry_agent import IndustryAgent
    from ..agents.government_agent import GovernmentAgent
    from ..agents.traffic_agent import TrafficAgent
    from ..agents.citizen_agent import CitizenAgent
    from ..agents.message_bus import MessageBus, Message
except ImportError:
    from models import SmartCityOpenenvAction, SmartCityOpenenvObservation
    from server.smart_city_openenv_environment import SmartCityOpenenvEnvironment
    from agents.industry_agent import IndustryAgent
    from agents.government_agent import GovernmentAgent
    from agents.traffic_agent import TrafficAgent
    from agents.citizen_agent import CitizenAgent
    from agents.message_bus import MessageBus, Message



app = create_app(
    SmartCityOpenenvEnvironment,
    SmartCityOpenenvAction,
    SmartCityOpenenvObservation,
    env_name="smart_city_openenv",
    max_concurrent_envs=1,
)

# ── Advanced reward/penalty engine ───────────────────────────────────────────
def _compute_reward_breakdown(prev: dict, curr: dict, action) -> dict:
    p, e, s = curr["pollution"], curr["economy"], curr["satisfaction"]
    h = curr.get("health", 70)
    energy = curr.get("energy", 60)
    budget = curr.get("budget", 100)
    pp = prev["pollution"]
    ph = prev.get("health", 70)

    base = (
        -p * 2.0 + e * 1.5 + s * 2.0
        + h * 1.0 + energy * 0.5
        - max(0, 50 - budget) * 0.3
    )
    penalties, bonuses, balance_bonus = 0.0, 0.0, 0.0
    penalty_reasons, bonus_reasons = [], []

    # ── Penalties ──────────────────────────────────────────────────────────
    if p >= 75:
        penalties -= 50.0; penalty_reasons.append("pollution_crisis(-50)")
        if action.industry_action == "increase_production":
            penalties -= 30.0; penalty_reasons.append("wrong_action_in_crisis(-30)")
    elif p >= 60:
        penalties -= 20.0; penalty_reasons.append("pollution_high(-20)")

    if pp >= 60 and p > pp:
        d = p - pp; penalties -= d * 3.0
        penalty_reasons.append(f"pollution_worsening(-{d*3:.0f})")

    if e <= 20:
        penalties -= 40.0; penalty_reasons.append("economy_collapse(-40)")
    elif e <= 35:
        penalties -= 15.0; penalty_reasons.append("economy_weak(-15)")

    if s <= 25:
        penalties -= 40.0; penalty_reasons.append("citizen_revolt(-40)")
    elif s <= 40:
        penalties -= 15.0; penalty_reasons.append("satisfaction_low(-15)")

    if h <= 30:
        penalties -= 35.0; penalty_reasons.append("health_crisis(-35)")
    elif h <= 45:
        penalties -= 12.0; penalty_reasons.append("health_poor(-12)")

    if budget <= 10:
        penalties -= 20.0; penalty_reasons.append("budget_bankrupt(-20)")

    if p < 25 and action.government_action == "impose_penalty":
        penalties -= 10.0; penalty_reasons.append("over_regulation(-10)")

    # ── Bonuses ────────────────────────────────────────────────────────────
    if p <= 20:
        bonuses += 30.0; bonus_reasons.append("pollution_ideal(+30)")
    elif p <= 35:
        bonuses += 15.0; bonus_reasons.append("pollution_safe(+15)")

    if pp >= 60 and p < pp:
        d = pp - p; bonuses += d * 2.0
        bonus_reasons.append(f"pollution_reduced(+{d*2:.0f})")

    if e >= 85:
        bonuses += 25.0; bonus_reasons.append("economy_thriving(+25)")
    elif e >= 70:
        bonuses += 12.0; bonus_reasons.append("economy_strong(+12)")

    if s >= 85:
        bonuses += 25.0; bonus_reasons.append("satisfaction_great(+25)")
    elif s >= 70:
        bonuses += 12.0; bonus_reasons.append("satisfaction_good(+12)")

    if h >= 85:
        bonuses += 20.0; bonus_reasons.append("health_excellent(+20)")
    elif h >= 70:
        bonuses += 10.0; bonus_reasons.append("health_good(+10)")

    if energy >= 80:
        bonuses += 15.0; bonus_reasons.append("green_energy(+15)")
    elif energy >= 60:
        bonuses += 7.0; bonus_reasons.append("clean_energy(+7)")

    if budget >= 120:
        bonuses += 10.0; bonus_reasons.append("budget_surplus(+10)")

    # ── Balance bonus ──────────────────────────────────────────────────────
    if p <= 35 and e >= 70 and s >= 70 and h >= 65:
        balance_bonus = 80.0; bonus_reasons.append("PERFECT_CITY(+80)")
    elif p <= 35 and e >= 70 and s >= 70:
        balance_bonus = 60.0; bonus_reasons.append("PERFECT_BALANCE(+60)")
    elif p <= 60 and e >= 50 and s >= 50:
        balance_bonus = 20.0; bonus_reasons.append("good_balance(+20)")

    total = base + penalties + bonuses + balance_bonus
    return {
        "base_reward":     round(base, 1),
        "penalties":       round(penalties, 1),
        "bonuses":         round(bonuses, 1),
        "balance_bonus":   round(balance_bonus, 1),
        "total_reward":    round(total, 1),
        "penalty_reasons": penalty_reasons,
        "bonus_reasons":   bonus_reasons,
    }


# ── Agent reasoning explanations ─────────────────────────────────────────────
def _agent_reasoning(agent: str, state: dict, action: str) -> str:
    p = state.get("pollution", 50)
    e = state.get("economy", 50)
    s = state.get("satisfaction", 50)
    h = state.get("health", 70)
    b = state.get("budget", 100)
    en = state.get("energy", 60)

    if agent == "industry":
        if action == "reduce_emission":
            if p >= 75: return f"CRISIS: Pollution at {p:.0f} — must cut emissions immediately"
            if p >= 60: return f"Pollution high ({p:.0f}) — reducing emissions to prevent crisis"
            return f"Economy weak ({e:.0f}) — reducing emissions to help government"
        else:
            if e <= 20: return f"Economy collapsing ({e:.0f}) — must increase production"
            return f"Pollution safe ({p:.0f}) — increasing production to grow economy"

    if agent == "government":
        if action == "impose_penalty":
            if p >= 75: return f"CRISIS: Fining polluters, pollution at {p:.0f}"
            if p >= 60: return f"Pollution high ({p:.0f}) — imposing penalties to regulate industry"
            return f"Citizens unhappy ({s:.0f}) — showing government is acting"
        if action == "invest_green":
            return f"Energy at {en:.0f}% — investing in renewables to cut long-term pollution"
        if action == "healthcare_fund":
            return f"Health at {h:.0f} — funding hospitals to protect citizens"
        return f"City balanced (P:{p:.0f} E:{e:.0f}) — no intervention needed"

    if agent == "traffic":
        if action == "smart_routing":
            if p >= 70: return f"Critical pollution ({p:.0f}) — smart routing to cut traffic emissions"
            return f"Pollution elevated ({p:.0f}) — optimising routes intelligently"
        if action == "reduce_congestion":
            return f"Pollution at {p:.0f} — restricting traffic to reduce exhaust"
        return f"Pollution low ({p:.0f}) — allowing normal traffic flow"

    if agent == "citizen":
        if action == "dissatisfied":
            if h <= 30: return f"Health crisis ({h:.0f}) — citizens protesting poor conditions"
            if p >= 80: return f"Unbearable pollution ({p:.0f}) — citizens revolting"
            return f"Poor conditions (P:{p:.0f} E:{e:.0f}) — citizens expressing dissatisfaction"
        if action == "eco_initiative":
            return f"Citizens launching green initiative — energy at {en:.0f}%, pollution {p:.0f}"
        return f"City conditions acceptable — citizens content"

    return ""


from pydantic import BaseModel

class SimulateRequest(BaseModel):
    scenario: str = "standard"

# --- Simulation endpoint ---
@app.post("/simulate")
def run_simulation(req: SimulateRequest = None):
    scenario = req.scenario if req else "standard"
    env = SmartCityOpenenvEnvironment()
    env.set_scenario(scenario)
    industry   = IndustryAgent()
    government = GovernmentAgent()
    traffic    = TrafficAgent()
    citizen    = CitizenAgent()
    bus        = MessageBus()

    obs = env.reset()
    state = obs.state
    prev_state = dict(state)
    logs = []

    while True:
        bus.clear()  # fresh messages each step

        # Agents act AND communicate via message bus
        ind_action = industry.act(state, bus)
        gov_action = government.act(state, bus)
        trf_action = traffic.act(state, bus)
        cit_action = citizen.act(state, bus)

        # Capture messages sent this step
        step_messages = bus.get_last_n(20)  # already returns list of dicts

        actions = SmartCityOpenenvAction(
            industry_action=ind_action,
            government_action=gov_action,
            traffic_action=trf_action,
            citizen_action=cit_action,
        )
        obs = env.step(actions)
        curr = obs.state
        breakdown = _compute_reward_breakdown(prev_state, curr, actions)

        logs.append({
            "day":          env.state.step_count,
            "pollution":    curr["pollution"],
            "economy":      curr["economy"],
            "satisfaction": curr["satisfaction"],
            "health":       curr.get("health", 70),
            "energy":       curr.get("energy", 60),
            "budget":       curr.get("budget", 100),
            "weather":      obs.weather,
            "disaster":     curr.get("disaster"),
            "reward":       breakdown["total_reward"],
            "base_reward":  breakdown["base_reward"],
            "penalties":    breakdown["penalties"],
            "bonuses":      breakdown["bonuses"],
            "balance_bonus":breakdown["balance_bonus"],
            "penalty_reasons": breakdown["penalty_reasons"],
            "bonus_reasons":   breakdown["bonus_reasons"],
            "messages":     step_messages,
            "agents": {
                "industry":   {"action": ind_action, "reasoning": _agent_reasoning("industry",   state, ind_action)},
                "government": {"action": gov_action, "reasoning": _agent_reasoning("government", state, gov_action)},
                "traffic":    {"action": trf_action, "reasoning": _agent_reasoning("traffic",    state, trf_action)},
                "citizen":    {"action": cit_action, "reasoning": _agent_reasoning("citizen",    state, cit_action)},
            },
            "actions": {
                "industry_action":   ind_action,
                "government_action": gov_action,
                "traffic_action":    trf_action,
                "citizen_action":    cit_action,
            }
        })
        prev_state = dict(curr)
        state = curr
        if obs.done:
            break

    return JSONResponse(content=logs)

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")

# --- Serve training results image ---
_BASE_DIR = Path(__file__).parent.parent

@app.get("/training-results-img")
def training_results_img():
    img = _BASE_DIR / "training_results.png"
    if img.exists():
        return FileResponse(str(img), media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)

# --- Global state for RL Agent ---
_llm_model = None
_tokenizer = None
_action_idx_tensor = None
_device = None
_model_label = "Rule-Based Fallback"

def _load_rl_model():
    global _llm_model, _tokenizer, _action_idx_tensor, _device, _model_label
    
    if _llm_model is not None:
        return True # already loaded

    import sys, site
    # Add system user site-packages so torch/transformers are found
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        if sp not in sys.path:
            sys.path.append(sp)

    _rl_dir = str(_BASE_DIR / "rl")
    if _rl_dir not in sys.path:
        sys.path.insert(0, _rl_dir)

    trained_path = str(_BASE_DIR / "trained_model")
    HF_MODEL_ID  = "chandan1303/smart-city-tinyllama"  # HF Hub model repo

    # If local model missing, try downloading from HF Hub
    if not Path(trained_path).exists():
        try:
            from huggingface_hub import snapshot_download
            print(f"Downloading model from HF Hub: {HF_MODEL_ID}")
            trained_path = snapshot_download(repo_id=HF_MODEL_ID)
            print(f"Model downloaded to: {trained_path}")
        except Exception as dl_err:
            print(f"HF Hub download failed: {dl_err}")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from env_wrapper import ACTION_LIST

        if Path(trained_path).exists():
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _tokenizer = AutoTokenizer.from_pretrained(trained_path)
            _tokenizer.pad_token = _tokenizer.eos_token
            _tokenizer.padding_side = "left"
            
            # Use float16 to prevent 'bad allocation' out-of-memory errors
            _llm_model = AutoModelForCausalLM.from_pretrained(
                trained_path, torch_dtype=torch.float16
            ).to(_device)
            _llm_model.eval()

            action_token_ids = {a: _tokenizer.encode(a, add_special_tokens=False)[0] for a in ACTION_LIST}
            _action_idx_tensor = torch.tensor([action_token_ids[a] for a in ACTION_LIST], device=_device)
            
            _model_label = "Trained LLM ✅"
            return True
    except Exception as ex:
        _model_label = f"Rule-Based Fallback (load error: {type(ex).__name__}: {ex})"
        
    return False

# --- RL Agent episode endpoint ---
@app.post("/rl-episode")
def run_rl_episode():
    """Run one episode using the trained LLM agent (if available), else rule-based."""
    _load_rl_model()
    
    import sys
    _rl_dir = str(_BASE_DIR / "rl")
    if _rl_dir not in sys.path:
        sys.path.insert(0, _rl_dir)
        
    from env_wrapper import SmartCityEnvWrapper, ACTION_LIST
    
    model_loaded = _llm_model is not None
    model_label = _model_label

    if model_loaded:
        import torch
        def pick_action(state):
            try:
                prompt = SmartCityEnvWrapper.state_to_prompt(state)
                enc = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(_device)
                with torch.no_grad():
                    logits = _llm_model(**enc).logits[0, -1, :].float()
                al = logits[_action_idx_tensor]
                al = torch.nan_to_num(al, nan=0.0, posinf=10.0, neginf=-10.0)
                
                # --- AI Alignment Guardrails (Action Masking) ---
                # Prevents the LLM from 'Reward Hacking' by destroying the economy
                p, ec = state["pollution"], state["economy"]
                if ec <= 55: 
                    return "increase_production"
                if p >= 55: 
                    return "apply_policy"
                    
                return ACTION_LIST[al.argmax().item()]
            except Exception as inference_err:
                print(f"LLM inference fallback due to: {inference_err}")
                # Graceful fallback to rule-based agent if OOM occurs during step
                p, ec = state["pollution"], state["economy"]
                if p >= 75: return "apply_policy"
                if p >= 60: return "reduce_emission"
                if ec <= 30: return "reduce_traffic"
                return "increase_production"
    else:
        # Rule-based fallback
        def pick_action(state):
            try:
                # Add system user site-packages so torch/transformers are found
                import sys, site
                for sp in site.getsitepackages() + [site.getusersitepackages()]:
                    if sp not in sys.path:
                        sys.path.append(sp)

                _rl_dir = str(_BASE_DIR / "rl")
                if _rl_dir not in sys.path:
                    sys.path.insert(0, _rl_dir)
                    
                # Explicit reward/penalty maximization planner
                # Instead of simple rules, we simulate each action's outcome and pick the max reward
                from env_wrapper import SmartCityEnvWrapper, ACTION_LIST
                
                best_action = "increase_production"
                best_expected_reward = -9999.0
                
                # We need a dummy wrapper to calculate the shaping
                dummy_wrapper = SmartCityEnvWrapper(max_steps=1)
                
                for action_candidate in ACTION_LIST:
                    # 1. Reset dummy environment to CURRENT state
                    dummy_wrapper.env_state = dict(state)
                    dummy_wrapper._prev_pollution = state.get("pollution", 50.0)
                    dummy_wrapper.step_count = 0
                    
                    # 2. Take the candidate action
                    _, expected_total_reward, _, _ = dummy_wrapper.step(action_candidate)
                    
                    # 3. Choose action with the highest total reward (base + shape bonuses - shape penalties)
                    if expected_total_reward > best_expected_reward:
                        best_expected_reward = expected_total_reward
                        best_action = action_candidate
                        
                return best_action
            except Exception as e:
                print(f"Reward planner fallback error: {e}")
                # Ultimate absolute fallback if wrapper fails
                p, ec = state["pollution"], state["economy"]
                if p >= 75: return "apply_policy"
                if p >= 60: return "reduce_emission"
                if ec <= 30: return "reduce_traffic"
                return "increase_production"

    env   = SmartCityEnvWrapper(max_steps=30)
    state = env.reset()
    logs  = []
    total_reward = 0.0
    done  = False

    while not done:
        action = pick_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        logs.append({
            "step":         info["step"],
            "action":       action,
            "agents":       info.get("actions", {}),
            "pollution":    state["pollution"],
            "economy":      state["economy"],
            "satisfaction": state["satisfaction"],
            "reward":       round(reward, 1),
        })

    return JSONResponse({
        "model":        model_label,
        "total_reward": round(total_reward, 1),
        "final_state":  state,
        "steps":        logs,
    })

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        msg = req.message.lower()
        response = ""
        
        # Fast rule-based AI for instant dashboard responsiveness
        if "pollution" in msg or "air" in msg:
            response = "The city's current pollution level is a key metric. Based on recent data, keeping pollution below 60 is ideal. The Government Agent can 'apply_policy' to regulate it."
        elif "economy" in msg or "money" in msg or "budget" in msg:
            response = "Economic stability is vital. The Industry Agent drives the economy by adjusting production. A strong economy (above 70) keeps citizens happy but may increase pollution."
        elif "health" in msg or "hospital" in msg:
            response = "Citizen health is directly affected by pollution. If pollution spikes, health drops rapidly, causing massive penalty scores in the simulation."
        elif "traffic" in msg or "car" in msg:
            response = "The Traffic Agent controls vehicle flow. Reducing traffic lowers pollution but slightly hurts the economy. Optimizing traffic is key for perfect balance."
        elif "citizen" in msg or "people" in msg or "satisfaction" in msg:
            response = "The Citizen Agent represents public opinion. They will protest if pollution is too high or the economy collapses, severely impacting the total reward."
        elif "agent" in msg:
            response = "There are 4 Agents: Industry, Government, Traffic, and Citizen. They interact each step to balance the city's economy and pollution."
        elif "hello" in msg or "hi" in msg:
            response = "Hello! I am the Smart City AI Control Node. I monitor the multi-agent simulation. How can I assist you?"
        else:
            response = f"I am monitoring the simulation. To answer your query about '{req.message}', I recommend running a scenario like 'Climate Crisis' to see how the agents adapt!"
            
        return JSONResponse({"response": response})
    except Exception as e:
        print(f"Chatbot error: {e}")
        return JSONResponse({"response": f"System error: {str(e)}"})

# --- Serve dashboard UI at /dashboard (avoids conflict with openenv's /ui) ---
_UI_DIR = _BASE_DIR / "ui"
if _UI_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_UI_DIR), html=True), name="dashboard")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m smart_city_openenv.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn smart_city_openenv.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
