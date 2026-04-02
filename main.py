from fastapi import FastAPI
from models import Action, Observation, StepResult
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


@app.get("/state")
def state():
    return env.state().model_dump()


@app.get("/tasks")
def tasks():
    return [
        "fault_diagnosis",
        "driving_decision",
        "autonomous_control",
    ]


@app.get("/schema")
def schema():
    return {
        "Observation": Observation.model_json_schema(),
        "Action": Action.model_json_schema(),
        "StepResult": StepResult.model_json_schema(),
    }