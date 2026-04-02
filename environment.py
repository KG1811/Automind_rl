# ==============================
# AutoMind OpenEnv - Environment (FINAL FIXED)
# ==============================

from __future__ import annotations

import random
from typing import Optional

from models import (
    Observation,
    FailureState,
    EpisodeState,
    Action,
    StepResult,
    Metrics,
)

from simulator import AutoMindSimulator


class AutoMindEnv:

    def __init__(self, seed: int = 42, max_steps: int = 20) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps

        self.simulator = AutoMindSimulator(seed=seed)

        self.current_task: Optional[str] = None
        self.current_difficulty: Optional[str] = None

        self.episode_state: EpisodeState = EpisodeState(max_steps=max_steps)
        self.current_observation: Optional[Observation] = None

        self.override_active = False

    # ---------------------------------
    # INITIAL STATE
    # ---------------------------------
    def _build_initial_observation(self, task_name: str, difficulty: str) -> Observation:

        if difficulty == "easy":
            return Observation(
                speed=35.0,
                engine_temp=88.0,
                distance_to_obstacle=60.0,
                road_condition="dry",
                oil_level=78.0,
                battery_health=84.0,
                failures=FailureState(),
                history=[],
            )

        if difficulty == "medium":
            return Observation(
                speed=48.0,
                engine_temp=98.0,
                distance_to_obstacle=28.0,
                road_condition="wet",
                oil_level=32.0,
                battery_health=70.0,
                failures=FailureState(low_oil=True),
                history=[],
            )

        if difficulty == "hard":
            return Observation(
                speed=62.0,
                engine_temp=114.0,
                distance_to_obstacle=14.0,
                road_condition="rain",
                oil_level=18.0,
                battery_health=61.0,
                failures=FailureState(
                    brake_failure=True,
                    sensor_failure=True,
                    engine_overheating=True,
                    low_oil=True,
                ),
                history=[],
            )

        raise ValueError("Invalid difficulty")

    # ---------------------------------
    # RESET
    # ---------------------------------
    def reset(self, task_name: str = "fault_diagnosis", difficulty: str = "easy") -> Observation:

        self.current_task = task_name
        self.current_difficulty = difficulty

        self.episode_state = EpisodeState(
            step_count=0,
            max_steps=self.max_steps,
            is_collision=False,
            is_engine_failure=False,
            is_safe_stop=False,
        )

        self.current_observation = self._build_initial_observation(task_name, difficulty)

        self.override_active = False

        return self.current_observation

    # ---------------------------------
    # STATE
    # ---------------------------------
    def state(self) -> Observation:
        if self.current_observation is None:
            raise RuntimeError("Call reset() first")
        return self.current_observation

    # ---------------------------------
    # STEP
    # ---------------------------------
    def step(self, action: Action) -> StepResult:

        if self.current_observation is None:
            raise RuntimeError("Call reset() before step()")

        if self.current_difficulty is None:
            raise RuntimeError("Call reset() first")

        if not (0.0 <= action.value <= 1.0):
            raise ValueError("Action value must be between 0 and 1")

        # ✅ Deterministic override
        if self.episode_state.step_count % 3 == 0:
            self.override_active = True
        else:
            self.override_active = False

        if self.override_active and action.action_type in ["stop", "brake"]:
            action = Action(
                action_type="accelerate",
                value=action.value,
                reason="Human override",
            )

        sim_result = self.simulator.transition(
            observation=self.current_observation,
            action_type=action.action_type,
            action_value=action.value,
            difficulty=self.current_difficulty,
        )

        new_observation = sim_result["observation"]

        # Update episode state
        self.episode_state.step_count += 1

        self.episode_state.is_collision = (
            self.episode_state.is_collision or sim_result["is_collision"]
        )

        self.episode_state.is_engine_failure = sim_result["is_engine_failure"]

        if action.action_type == "stop" and new_observation.speed <= 3:
            self.episode_state.is_safe_stop = True

        # ✅ Correct reward (uses new state)
        reward = self._compute_reward(action, sim_result, new_observation)

        done = self._check_done()

        # ✅ Correct metrics (uses new state)
        metrics = self._compute_metrics(sim_result, new_observation)

        self.current_observation = new_observation

        return StepResult(
            observation=new_observation,
            reward=reward,
            done=done,
            info={
                "outcome": self.get_episode_outcome(),
                "collision_risk": sim_result["collision_risk"],
                "override_active": self.override_active,
            },
            metrics=metrics,
        )

    # ---------------------------------
    # REWARD
    # ---------------------------------
    def _compute_reward(self, action: Action, sim_result: dict, new_observation: Observation) -> float:

        reward = 0.0

        if sim_result["is_collision"]:
            return -1.0

        safety_score = 1.0 - sim_result["collision_risk"]
        reward += 0.4 * safety_score

        efficiency_score = max(0.0, 1.0 - sim_result["collision_risk"])
        reward += 0.2 * efficiency_score

        if new_observation.failures.engine_overheating:
            if action.action_type in ["stop", "request_service"]:
                reward += 0.2

        if new_observation.failures.low_oil:
            if action.action_type in ["stop", "request_service"]:
                reward += 0.2

        if action.action_type == "brake" and sim_result["collision_risk"] > 0.5:
            reward += 0.2

        if sim_result["collision_risk"] > 0.7 and action.action_type == "accelerate":
            reward -= 0.7

        if len(new_observation.history) >= 2:
            last_actions = [h.action_taken for h in new_observation.history[-2:]]
            if all(a == action.action_type for a in last_actions):
                reward -= 0.1

        if self.override_active:
            reward -= 0.2

        if self.override_active and action.action_type in ["brake", "stop"]:
            reward += 0.3

        return max(-1.0, min(1.0, reward))

    # ---------------------------------
    # METRICS (FINAL FIXED)
    # ---------------------------------
    def _compute_metrics(self, sim_result: dict, new_observation: Observation) -> Metrics:

        safety_score = 1.0 if not sim_result["is_collision"] else 0.0
        efficiency_score = max(0.0, 1.0 - sim_result["collision_risk"])

        failures = new_observation.failures

        # 🔥 Better diagnosis realism
        if (
            failures.engine_overheating
            or failures.low_oil
            or failures.battery_issue
        ):
            diagnosis_score = 1.0
        else:
            diagnosis_score = 0.5

        # 🔥 Sequence awareness
        if self.episode_state.step_count < self.max_steps / 2:
            sequence_score = 1.0
        else:
            sequence_score = 0.7

        return Metrics(
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            diagnosis_score=diagnosis_score,
            sequence_score=sequence_score,
        )

    # ---------------------------------
    # DONE
    # ---------------------------------
    def _check_done(self) -> bool:
        return (
            self.episode_state.is_collision
            or self.episode_state.is_engine_failure
            or self.episode_state.is_safe_stop
            or self.episode_state.step_count >= self.max_steps
        )

    # ---------------------------------
    # OUTCOME
    # ---------------------------------
    def get_episode_outcome(self) -> str:

        if self.episode_state.is_collision:
            return "failure_collision"

        if self.episode_state.is_engine_failure:
            return "failure_engine"

        if self.episode_state.is_safe_stop:
            return "success_safe_stop"

        if self.episode_state.step_count >= self.max_steps:
            return "timeout"

        return "in_progress"