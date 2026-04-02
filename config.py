# ==============================
# AutoMind OpenEnv Configuration
# ==============================

ENV_NAME = "automind-openenv"

# ------------------------------
# Observation Fields (State Space)
# ------------------------------
OBSERVATION_FIELDS = [
    "speed",
    "engine_temp",
    "distance_to_obstacle",
    "road_condition",
    "oil_level",
    "battery_health",
    "failures",
    "history"
]

# ------------------------------
# Action Space
# ------------------------------
ACTION_SPACE = [
    "brake",
    "accelerate",
    "turn_left",
    "turn_right",
    "continue",
    "stop",
    "request_service"
]

# ------------------------------
# Failure Types
# ------------------------------
FAILURE_TYPES = [
    "brake_failure",
    "sensor_failure",
    "engine_overheating",
    "low_oil",
    "battery_issue"
]

# ------------------------------
# Task Definitions
# ------------------------------
TASKS = {
    "fault_diagnosis": {
        "type": "classification",
        "goal": "Detect vehicle fault correctly"
    },
    "driving_decision": {
        "type": "control",
        "goal": "Choose safe driving action"
    },
    "autonomous_control": {
        "type": "multi-step",
        "goal": "Full decision + diagnosis + safety handling"
    }
}

# ------------------------------
# Episode Design (CRITICAL FIX)
# ------------------------------
MAX_STEPS = 20

SUCCESS_CONDITIONS = [
    "safe_stop",
    "correct_service_decision"
]

FAILURE_CONDITIONS = [
    "collision",
    "engine_failure",
    "wrong_decision_sequence"
]

# ------------------------------
# Reward Weights (NOT GRADER)
# ------------------------------
REWARD_WEIGHTS = {
    "safety": 0.4,
    "diagnosis": 0.2,
    "efficiency": 0.2,
    "sequence": 0.2
}