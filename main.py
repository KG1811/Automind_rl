from fastapi import FastAPI
from models import Action
from environment import AutoMindEnv

app = FastAPI()
env = AutoMindEnv()


@app.get("/")
def health():
    return {"status": "AutoMind OpenEnv running"}


@app.post("/reset")
def reset(task_name: str = "fault_diagnosis", difficulty: str = "easy"):
    obs = env.reset(task_name, difficulty)
    return {"observation": obs.model_dump()}


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


# 🔥 FIXED STATE ENDPOINT
@app.get("/state")
def state():
    try:
        return env.get_full_state()
    except Exception as e:
        return {"error": str(e)}


@app.get("/tasks")
def tasks():
    return ["fault_diagnosis", "driving_decision", "autonomous_control"]


@app.get("/schema")
def schema():
    from models import Observation, Action, StepResult
    return {
        "Observation": Observation.model_json_schema(),
        "Action": Action.model_json_schema(),
        "StepResult": StepResult.model_json_schema(),
    }