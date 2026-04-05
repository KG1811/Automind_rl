# ==============================
# AutoMind Agent (FINAL UPGRADED)
# ==============================

from models import Action


# ---------------------------------
# HELPER: TEMPERATURE TREND
# ---------------------------------
def get_temp_trend(history):
    if len(history) < 3:
        return "stable"

    temps = [h.state_summary.get("engine_temp", 0) for h in history[-3:]]

    if temps[2] > temps[1] > temps[0]:
        return "rising"

    if temps[2] < temps[1] < temps[0]:
        return "falling"

    return "stable"


# ---------------------------------
# FAULT DIAGNOSIS (UPGRADED)
# ---------------------------------
def diagnose_fault(obs):
    """
    Multi-factor realistic diagnosis
    """

    # 🔥 Combined logic (better than simple threshold)
    if obs.engine_temp > 105 and obs.oil_level < 30:
        return "engine_overheating", "high"

    if obs.engine_temp > 110:
        return "engine_overheating", "high"

    if obs.oil_level < 25:
        return "low_oil", "medium"

    if obs.battery_health < 40:
        return "battery_issue", "medium"

    return "no_fault", "low"


# ---------------------------------
# DECISION POLICY (UPGRADED)
# ---------------------------------
def decide_action(obs, fault, urgency):

    temp_trend = get_temp_trend(obs.history)

    # 🚨 1. ESCALATION LOGIC
    if temp_trend == "rising" and obs.engine_temp > 105:
        return "stop", 0.9, "temperature rising consistently → preventive stop"

    # ⚠️ 2. FAULT HANDLING
    if fault == "engine_overheating" and urgency == "high":
        return "stop", 1.0, f"critical overheating ({obs.engine_temp})"

    if fault == "low_oil":
        return "stop", 0.7, "low oil detected"
        
    if fault == "battery_issue":
        return "stop", 0.8, "critical battery issue"

    # 🚗 3. NORMAL DRIVING / SAFELY CONTINUING
    if obs.speed > 55:
        return "continue", 0.4, "cruising safely at higher speeds"
        
    if obs.speed < 40:
        return "accelerate", 0.6, "safe acceleration"

    return "continue", 0.5, "stable driving"


# ---------------------------------
# MAIN AGENT
# ---------------------------------
def agent_step(observation, task_name="autonomous_control"):

    fault, urgency = diagnose_fault(observation)

    if task_name == "fault_diagnosis":
        return Action(action_type="diagnose", value=1.0, reason=fault)

    action_type, value, reason = decide_action(
        observation, fault, urgency
    )

    return Action(
        action_type=action_type,
        value=value,
        reason=reason,
    )