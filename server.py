# ==============================
# AutoMind OpenEnv - API Server
# Phase 10: OpenEnv Interface
# ==============================

from flask import Flask, request, jsonify

from environment import AutoMindEnv
from models import Action

app = Flask(__name__)

env = AutoMindEnv()


# ---------------------------------
# RESET
# ---------------------------------
@app.route("/reset", methods=["POST"])
def reset():

    data = request.json or {}

    task = data.get("task_name", "fault_diagnosis")
    difficulty = data.get("difficulty", "easy")

    obs = env.reset(task_name=task, difficulty=difficulty)

    return jsonify({
        "observation": obs.model_dump()
    })


# ---------------------------------
# STEP
# ---------------------------------
@app.route("/step", methods=["POST"])
def step():

    data = request.json

    if not data:
        return jsonify({"error": "No input provided"}), 400

    try:
        action = Action(
            action_type=data["action_type"],
            value=float(data["value"]),
            reason=data.get("reason", "")
        )

        result = env.step(action)

        return jsonify({
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
            "metrics": result.metrics.model_dump()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# STATE
# ---------------------------------
@app.route("/state", methods=["GET"])
def state():

    try:
        return jsonify(env.state().model_dump())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# HEALTH CHECK
# ---------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AutoMind OpenEnv running"})


# ---------------------------------
# RUN SERVER
# ---------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)