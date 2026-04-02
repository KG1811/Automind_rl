# ==============================
# AutoMind OpenEnv - Inference Script (FINAL)
# ==============================

import os
import requests

from models import Action
from agent import agent_step
from tasks import evaluate_task

# ------------------------------
# ENV CONFIG (MANDATORY)
# ------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "automind-agent")  # not used but required
HF_TOKEN = os.getenv("HF_TOKEN", "dummy")  # not used but required

MAX_STEPS = 20


# ------------------------------
# HELPER: PRINT STATE
# ------------------------------
def print_state(step, obs):
    print(f"\nSTEP {step}")
    print("STATE:")
    print(f"  Speed: {obs['speed']}")
    print(f"  Distance: {obs['distance_to_obstacle']}")
    print(f"  Temp: {obs['engine_temp']}")
    print(f"  Oil: {obs['oil_level']}")
    print(f"  Battery: {obs['battery_health']}")


# ------------------------------
# MAIN RUN FUNCTION
# ------------------------------
def run_episode(task_name="autonomous_control", difficulty="medium"):

    print("\n" + "=" * 50)
    print(f"RUNNING TASK: {task_name.upper()} | DIFFICULTY: {difficulty.upper()}")
    print("=" * 50)

    # --------------------------
    # RESET
    # --------------------------
    response = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_name": task_name, "difficulty": difficulty},
    )

    data = response.json()
    obs = data["observation"]

    last_action = None
    last_metrics = None

    # --------------------------
    # LOOP
    # --------------------------
    for step in range(1, MAX_STEPS + 1):

        print_state(step, obs)

        # Convert dict → object for agent
        from models import Observation
        observation_obj = Observation(**obs)

        # ----------------------
        # AGENT STEP
        # ----------------------
        action = agent_step(observation_obj)

        print("\nACTION:")
        print(f"  {action.action_type} ({action.value})")
        print(f"  Reason: {action.reason}")

        # ----------------------
        # CALL STEP API
        # ----------------------
        response = requests.post(
            f"{API_BASE_URL}/step",
            json={
                "action_type": action.action_type,
                "value": action.value,
                "reason": action.reason,
            },
        )

        result = response.json()

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        metrics = result["metrics"]

        print("\nRESULT:")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")

        print("\nMETRICS:")
        print(metrics)

        print("\nINFO:")
        print(result["info"])

        last_action = action
        last_metrics = metrics

        if done:
            print("\nEPISODE FINISHED")
            break

    # --------------------------
    # FINAL SCORE (CRITICAL)
    # --------------------------
    final_score = evaluate_task(
        task_name=task_name,
        action=last_action,
        observation=observation_obj,
        metrics=last_metrics,
    )

    print("\n" + "=" * 50)
    print("FINAL SCORE:", final_score)
    print("=" * 50)

    return final_score


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":

    scores = []

    for difficulty in ["easy", "medium", "hard"]:
        score = run_episode(
            task_name="autonomous_control",
            difficulty=difficulty,
        )
        scores.append(score)

    print("\n" + "=" * 50)
    print("AVERAGE SCORE:", sum(scores) / len(scores))
    print("=" * 50)