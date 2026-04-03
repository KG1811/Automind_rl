from fastapi import FastAPI
from models import Action, Observation
from environment import AutoMindEnv

app = FastAPI()

env = AutoMindEnv()


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
        "observation": obs.model_dump()   # ✅ FIXED
    }


# ---------------------------------
# STEP
# ---------------------------------
@app.post("/step")
def step(action: Action):
    result = env.step(action)

    return {
        "observation": result.observation.model_dump(),  # ✅ FIXED
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
        "metrics": result.metrics.model_dump(),          # ✅ FIXED
    }


# ---------------------------------
# STATE
# ---------------------------------
@app.get("/state")
def state():
    obs = env.state()
    return obs.model_dump()   # ✅ FIXED


# ---------------------------------
# TASKS
# ---------------------------------
@app.get("/tasks")
def tasks():
    return [
        "fault_diagnosis",
        "driving_decision",
        "autonomous_control",   # ✅ FIXED NAME
    ]


# ---------------------------------
# SCHEMA (IMPORTANT FOR OPENENV)
# ---------------------------------
@app.get("/schema")
def schema():
    return {
        "Observation": Observation.model_json_schema(),  # ✅ REAL SCHEMA
        "Action": Action.model_json_schema(),
    }


# ---------------------------------
# RUN SERVER
# ---------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)