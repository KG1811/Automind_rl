# ==============================
# TEST AGENT (FINAL)
# ==============================

from environment import AutoMindEnv
from agent import agent_step


def get_temp_trend(history):
    if len(history) < 3:
        return "stable"

    temps = [h.state_summary.get("engine_temp", 0) for h in history[-3:]]

    if temps[2] > temps[1] > temps[0]:
        return "rising"
    return "stable"


def run_test(difficulty):

    print("\n" + "=" * 50)
    print(f"TESTING: AUTONOMOUS_CONTROL | DIFFICULTY: {difficulty.upper()}")
    print("=" * 50)

    env = AutoMindEnv()
    obs = env.reset("autonomous_control", difficulty)

    for i in range(20):

        print(f"\nSTEP {i+1}")
        print("STATE:")
        print(f"  Speed: {obs.speed}")
        print(f"  Distance: {obs.distance_to_obstacle}")
        print(f"  Temp: {obs.engine_temp}")
        print(f"  Oil: {obs.oil_level}")
        print(f"  Battery: {obs.battery_health}")

        trend = get_temp_trend(obs.history)
        if trend == "rising":
            print("  Temp Trend: rising")

        action = agent_step(obs)

        print("\nACTION:")
        print(f"  Action: {action.action_type} ({action.value})")
        print(f"  Reason: {action.reason}")

        result = env.step(action)

        print("\nRESULT:")
        print(f"  Reward: {result.reward}")
        print(f"  Done: {result.done}")

        print("\nMETRICS:")
        print(result.metrics.model_dump())

        print("\nINFO:")
        print(result.info)

        obs = result.observation

        if result.done:
            print("\nEPISODE FINISHED")
            break

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print("Final Outcome:", result.info["outcome"])
    print("Final Metrics:")
    print(result.metrics.model_dump())
   


if __name__ == "__main__":
    for d in ["easy", "medium", "hard"]:
        run_test(d)