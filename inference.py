import json
import os
from typing import Optional

import requests
from openai import OpenAI

from agent import agent_step
from models import Action, Observation, Metrics
from tasks import evaluate_task

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_STEPS = 20
TEMPERATURE = 0.1




def build_prompt(observation: dict) -> str:
    return f"""
You are controlling an automotive agent.
Return JSON only in this format:
{{
  "action_type": "brake",
  "value": 0.7,
  "reason": "short reason"
}}

Observation:
{json.dumps(observation, indent=2)}

Choose one of:
brake, accelerate, turn_left, turn_right, continue, stop, request_service
""".strip()


def get_model_action(observation: dict) -> Optional[Action]:
    if not HF_TOKEN:
        return None

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(observation)}],
            temperature=TEMPERATURE,
            max_tokens=120,
        )

        text = completion.choices[0].message.content or ""
        payload = json.loads(text)

        return Action(
            action_type=payload["action_type"],
            value=float(payload["value"]),
            reason=payload["reason"],
        )
    except Exception:
        return None


def run_episode(task_name: str = "autonomous_control", difficulty: str = "medium") -> float:
    print("[START]")
    print(json.dumps({"task": task_name, "difficulty": difficulty}))

    response = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_name": task_name, "difficulty": difficulty},
        timeout=30,
    )
    response.raise_for_status()

    obs = response.json()["observation"]

    last_action = None
    last_metrics = None
    final_observation = None

    for step_idx in range(1, MAX_STEPS + 1):
        llm_action = get_model_action(obs)
        observation_obj = Observation(**obs)

        if llm_action is not None:
            action = llm_action
        else:
            action = agent_step(observation_obj)

        step_response = requests.post(
            f"{API_BASE_URL}/step",
            json=action.model_dump(),
            timeout=30,
        )
        step_response.raise_for_status()

        result = step_response.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        metrics = result["metrics"]

        print("[STEP]")
        print(json.dumps({
            "observation": obs,
            "action": action.model_dump(),
            "reward": reward,
            "done": done,
            "info": result["info"]
        }))

        last_action = action
        last_metrics = Metrics(**metrics)
        final_observation = Observation(**obs)

        if done:
            break

    final_score = evaluate_task(
        task_name=task_name,
        action=last_action,
        observation=final_observation or Observation(**obs),
        metrics=last_metrics,
    )

    print("[END]")
    print(json.dumps({"final_score": final_score}))
    return final_score


if __name__ == "__main__":
    scores = []
    for difficulty in ["easy", "medium", "hard"]:
        scores.append(run_episode(task_name="autonomous_control", difficulty=difficulty))

    pass