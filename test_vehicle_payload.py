from environment import AutoMindEnv


def test_hard_difficulty_exposes_vehicle_side_signals_and_events():
    env = AutoMindEnv()
    observation = env.reset("autonomous_control", "hard")

    assert observation.vehicle_signals.coolant_temp == observation.engine_temp
    assert observation.vehicle_signals.battery_voltage > 0.0
    assert 0.0 <= observation.vehicle_signals.fuel_level <= 100.0

    events = observation.vehicle_events
    assert events.mil_status is True
    assert events.dtc_count >= 2
    assert events.engine_overheat_warning is True
    assert events.low_oil_warning is True
