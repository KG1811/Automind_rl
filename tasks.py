# ==============================
# AutoMind OpenEnv - Tasks & Graders (Phase 7 FINAL)
# ==============================

from __future__ import annotations

from typing import Optional
from models import Observation, Action, Metrics


# =====================================
# TASK CONFIG
# =====================================

TASK_CONFIG = {
    "fault_diagnosis": {
        "allowed_actions": ["diagnose"],
        "goal": "Identify vehicle fault correctly",
    },
    "driving_decision": {
        "allowed_actions": [
            "brake",
            "accelerate",
            "turn_left",
            "turn_right",
            "continue",
            "stop",
        ],
        "goal": "Choose safest immediate driving action",
    },
    "autonomous_control": {
        "allowed_actions": [
            "brake",
            "accelerate",
            "turn_left",
            "turn_right",
            "continue",
            "stop",
            "request_service",
        ],
        "goal": "Full control with safety + diagnosis + efficiency",
    },
}


# =====================================
# TASK 1 — FAULT DIAGNOSIS
# =====================================

def detect_true_fault(observation: Observation) -> str:
    """
    Deterministic ground truth fault detection.
    """

    if observation.engine_temp >= 105:
        return "engine_overheating"

    if observation.oil_level <= 25:
        return "low_oil"

    if observation.battery_health <= 20:
        return "battery_issue"

    return "no_fault"


def grade_fault_diagnosis(predicted_fault: str, observation: Observation) -> float:
    """
    Strict deterministic scoring.
    """

    true_fault = detect_true_fault(observation)

    if predicted_fault == true_fault:
        return 1.0

    if true_fault != "no_fault" and predicted_fault != "no_fault":
        return 0.5

    return 0.0


# =====================================
# TASK 2 — DRIVING DECISION
# =====================================

def get_safe_action(observation: Observation) -> str:
    """
    Deterministic safe driving policy.
    """

    if observation.distance_to_obstacle < 10:
        return "brake"

    if observation.engine_temp > 105:
        return "stop"

    if observation.distance_to_obstacle > 40:
        return "accelerate"

    return "continue"


def grade_driving_decision(action: Action, observation: Observation) -> float:
    """
    Deterministic grading with partial credit.
    """

    correct_action = get_safe_action(observation)

    if action.action_type == correct_action:
        return 1.0

    if correct_action == "brake" and action.action_type == "stop":
        return 0.7

    if correct_action == "continue" and action.action_type == "accelerate":
        return 0.6

    if correct_action == "accelerate" and action.action_type == "continue":
        return 0.6

    return 0.0


# =====================================
# TASK 3 — FULL AUTONOMOUS CONTROL
# =====================================

def grade_autonomous_control(metrics: Metrics) -> float:
    """
    Final weighted deterministic score.
    """

    score = (
        0.4 * metrics.safety_score +
        0.2 * metrics.diagnosis_score +
        0.2 * metrics.efficiency_score +
        0.2 * metrics.sequence_score
    )

    return round(max(0.0, min(1.0, score)), 3)


# =====================================
# MASTER EVALUATION FUNCTION
# =====================================

def evaluate_task(
    task_name: str,
    action: Optional[Action],
    observation: Observation,
    metrics: Optional[Metrics],
    predicted_fault: Optional[str] = None,
) -> float:

    if task_name == "fault_diagnosis":
        if predicted_fault is None:
            return 0.0
        return grade_fault_diagnosis(predicted_fault, observation)

    if task_name == "driving_decision":
        if action is None:
            return 0.0
        return grade_driving_decision(action, observation)

    if task_name == "autonomous_control":
        if metrics is None:
            return 0.0
        return grade_autonomous_control(metrics)

    raise ValueError(f"Unknown task: {task_name}")