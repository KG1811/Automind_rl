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

    # 🚨 1. IMMEDIATE COLLISION SAFETY
    if obs.distance_to_obstacle < 5:
        return "brake", 1.0, f"extreme emergency: obstacle {obs.distance_to_obstacle}"

    if obs.distance_to_obstacle < 10:
        return "brake", 0.9, f"obstacle very close ({obs.distance_to_obstacle})"

    # 🔥 2. ESCALATION LOGIC (NEW)
    if temp_trend == "rising" and obs.engine_temp > 105:
        return "stop", 0.9, "temperature rising consistently → preventive stop"

    # ⚠️ 3. FAULT HANDLING
    if fault == "engine_overheating" and urgency == "high":
        return "stop", 1.0, f"critical overheating ({obs.engine_temp})"

    if fault == "low_oil":
        return "stop", 0.7, "low oil detected"

    # 🚗 4. SMART BRAKING (not binary)
    if obs.distance_to_obstacle < 20:
        if obs.speed > 40:
            return "brake", 0.9, "high speed + medium distance → preemptive braking"
        return "brake", 0.6, "moderate risk"

    # 🚗 5. NORMAL DRIVING
    if obs.speed < 40 and obs.distance_to_obstacle > 30:
        return "accelerate", 0.6, "safe acceleration"

    return "continue", 0.5, "stable driving"


# ---------------------------------
# MAIN AGENT
# ---------------------------------
def agent_step(observation):

    fault, urgency = diagnose_fault(observation)

    action_type, value, reason = decide_action(
        observation, fault, urgency
    )

    return Action(
        action_type=action_type,
        value=value,
        reason=reason,
    )