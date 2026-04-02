# ==============================
# AutoMind OpenEnv - Simulator Core
# ==============================

from __future__ import annotations

import random

from models import Observation, HistoryItem
from vehicle_dynamics import (
    apply_speed_decay,
    apply_action_to_speed,
    update_distance_to_obstacle,
    update_engine_temperature,
    estimate_collision_risk,
)
from failure_engine import (
    update_oil_level,
    update_battery_health,
    infer_failure_state,
    is_engine_failure,
)
from traffic_engine import (
    get_obstacle_relative_motion,
    get_traffic_pressure,
)
from noise_engine import (
    add_sensor_noise,
    maybe_corrupt_distance,
)


class AutoMindSimulator:
    """
    Core world update engine.

    This module updates:
    - speed
    - obstacle distance
    - temperature
    - oil
    - battery
    - failure evolution
    - collision risk

    It does NOT compute:
    - reward
    - grader score
    - API response routing
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def transition(
        self,
        observation: Observation,
        action_type: str,
        action_value: float,
        difficulty: str,
    ) -> dict:
        """
        Returns raw transition output.
        Environment will later wrap this into StepResult.
        """

        base_speed = apply_speed_decay(observation.speed)

        speed = apply_action_to_speed(
            speed=base_speed,
            action_type=action_type,
            action_value=action_value,
            road_condition=observation.road_condition,
            brake_failure=observation.failures.brake_failure,
        )

        obstacle_relative_motion = get_obstacle_relative_motion(
            rng=self.rng,
            difficulty=difficulty,
        )

        traffic_pressure = get_traffic_pressure(
            rng=self.rng,
            difficulty=difficulty,
        )

        distance_to_obstacle = update_distance_to_obstacle(
            current_distance=observation.distance_to_obstacle,
            speed=speed,
            obstacle_relative_motion=obstacle_relative_motion,
        )

        # Traffic pressure can effectively reduce safe clearance
        distance_to_obstacle = max(0.0, distance_to_obstacle - (traffic_pressure * 2.0))

        engine_temp = update_engine_temperature(
            engine_temp=observation.engine_temp,
            speed=speed,
            action_type=action_type,
            road_condition=observation.road_condition,
            overheating_active=observation.failures.engine_overheating,
        )

        oil_level = update_oil_level(
            oil_level=observation.oil_level,
            speed=speed,
            engine_temp=engine_temp,
            low_oil_active=observation.failures.low_oil,
        )

        battery_health = update_battery_health(
            battery_health=observation.battery_health,
            action_type=action_type,
            battery_issue_active=observation.failures.battery_issue,
        )

        failures = infer_failure_state(
            current_failures=observation.failures,
            engine_temp=engine_temp,
            oil_level=oil_level,
            battery_health=battery_health,
        )

        collision_risk = estimate_collision_risk(
            speed=speed,
            distance_to_obstacle=distance_to_obstacle,
            road_condition=observation.road_condition,
            brake_failure=failures.brake_failure,
        )

        is_collision = distance_to_obstacle <= 0.0 or collision_risk >= 0.95
        engine_failure = is_engine_failure(
            engine_temp=engine_temp,
            oil_level=oil_level,
        )

        observed_speed = add_sensor_noise(
            rng=self.rng,
            value=speed,
            std_dev=1.0,
            low=0.0,
            high=200.0,
        )

        observed_engine_temp = add_sensor_noise(
            rng=self.rng,
            value=engine_temp,
            std_dev=1.2,
            low=0.0,
            high=150.0,
        )

        observed_distance = maybe_corrupt_distance(
            rng=self.rng,
            value=distance_to_obstacle,
            sensor_failure=failures.sensor_failure,
        )

        updated_history = list(observation.history)
        updated_history.append(
            HistoryItem(
                state_summary={
                    "speed": round(observation.speed, 2),
                    "engine_temp": round(observation.engine_temp, 2),
                    "distance_to_obstacle": round(observation.distance_to_obstacle, 2),
                },
                action_taken=action_type,
            )
        )

        new_observation = Observation(
            speed=round(observed_speed, 2),
            engine_temp=round(observed_engine_temp, 2),
            distance_to_obstacle=round(observed_distance, 2),
            road_condition=observation.road_condition,
            oil_level=round(oil_level, 2),
            battery_health=round(battery_health, 2),
            failures=failures,
            history=updated_history[-5:],
        )

        return {
            "observation": new_observation,
            "collision_risk": round(collision_risk, 3),
            "traffic_pressure": round(traffic_pressure, 3),
            "is_collision": is_collision,
            "is_engine_failure": engine_failure,
        }


if __name__ == "__main__":
    from models import FailureState

    sim = AutoMindSimulator(seed=42)

    obs = Observation(
        speed=48.0,
        engine_temp=97.0,
        distance_to_obstacle=25.0,
        road_condition="wet",
        oil_level=35.0,
        battery_health=78.0,
        failures=FailureState(),
        history=[],
    )

    result = sim.transition(
        observation=obs,
        action_type="brake",
        action_value=0.8,
        difficulty="medium",
    )

    print("SIMULATOR TEST OK")
    print(result["observation"].model_dump())
    print(
        {
            "collision_risk": result["collision_risk"],
            "traffic_pressure": result["traffic_pressure"],
            "is_collision": result["is_collision"],
            "is_engine_failure": result["is_engine_failure"],
        }
    )