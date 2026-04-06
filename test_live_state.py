from environment import AutoMindEnv


def test_get_full_state_advances_after_update_interval():
    env = AutoMindEnv(update_interval_seconds=20)
    initial_observation = env.reset("autonomous_control", "easy")

    env.last_background_sync_at -= 20.1

    state = env.get_full_state()
    updated_observation = state["observation"]

    assert updated_observation["speed"] != initial_observation.speed
    assert updated_observation["engine_temp"] != initial_observation.engine_temp
    assert updated_observation["battery_health"] != initial_observation.battery_health
