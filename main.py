from fastapi import FastAPI
from models import Action, Observation
from environment import AutoMindEnv

import threading
import time

app = FastAPI()

env = AutoMindEnv()

# ---------------------------------
# AUTO SIMULATION LOOP (NEW 🔥)
# ---------------------------------
def auto_simulation():
    while True:
        try:
            if env.current_observation is not None:
                # default safe action (can later plug agent here)
                action = Action(
                    action_type="continue",
                    value=0.5,
                    reason="auto simulation"
                )
                env.step(action)
        except Exception as e:
            print("Auto simulation error:", e)

        time.sleep(10)  # ⏱ every 10 seconds


# Start background thread
threading.Thread(target=auto_simulation, daemon=True).start()


# ---------------------------------
# HEALTH
# ---------------------------------
@app.get("/")
def health():
    return {"status": "AutoMind OpenEnv running"}


# ---------------------------------
# RESET
# ---------------------------------
@app.post("/reset")
def reset(task_name: str = "fault_diagnosis", difficulty: str = "easy"):
    obs = env.reset(task_name, difficulty)
    return {
        "observation": obs.model_dump()
    }


# ---------------------------------
# STEP (MANUAL CONTROL)
# ---------------------------------
@app.post("/step")
def step(action: Action):
    result = env.step(action)

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
        "metrics": result.metrics.model_dump(),
    }


# ---------------------------------
# STATE (AUTO UPDATING)
# ---------------------------------
@app.get("/state")
def state():
    obs = env.state()
    return obs.model_dump()


# ---------------------------------
# TASKS
# ---------------------------------
@app.get("/tasks")
def tasks():
    return [
        "fault_diagnosis",
        "driving_decision",
        "autonomous_control",
    ]


# ---------------------------------
# SCHEMA
# ---------------------------------
@app.get("/schema")
def schema():
    return {
        "Observation": Observation.model_json_schema(),
        "Action": Action.model_json_schema(),
    }


# ---------------------------------
# RUN SERVER
# ---------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)