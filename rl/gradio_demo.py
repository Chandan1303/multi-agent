"""
Gradio demo — runs a single simulation episode using the trained (or base) model.
Deploy on HF Spaces: set TRAINED=True to use saved model, False for base model.
Run: python rl/gradio_demo.py
"""
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from env_wrapper import ACTION_LIST
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from env_wrapper import SmartCityEnvWrapper

TRAINED    = os.path.exists("trained_model")
MODEL_PATH = "trained_model" if TRAINED else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def run_episode():
    env = SmartCityEnvWrapper(max_steps=30)
    state = env.reset()
    logs, total_reward = [], 0.0

    while True:
        prompt = SmartCityEnvWrapper.state_to_prompt(state)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=16, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = SmartCityEnvWrapper.parse_action(response)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        logs.append(
            f"Step {info['step']:2d} | Action: {action:22s} | "
            f"Pollution: {next_state['pollution']:5.1f} | "
            f"Economy: {next_state['economy']:5.1f} | "
            f"Reward: {reward:7.1f}"
        )
        state = next_state
        if done:
            break

    logs.append(f"\n{'='*60}")
    logs.append(f"Total Reward: {total_reward:.1f}")
    logs.append(f"Final Pollution: {state['pollution']:.1f}")
    logs.append(f"Model: {'Trained ✅' if TRAINED else 'Base (untrained) ⚠️'}")
    return "\n".join(logs)


demo = gr.Interface(
    fn=run_episode,
    inputs=[],
    outputs=gr.Textbox(label="Simulation Log", lines=35),
    title="🏙️ Smart City RL Agent Demo",
    description=(
        "Runs a 30-step Smart City simulation using an LLM agent. "
        "The agent chooses actions based on pollution, economy, and satisfaction levels."
    ),
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=True)
