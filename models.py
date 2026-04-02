# ==============================
# AutoMind OpenEnv - Typed Models
# ==============================

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# ------------------------------
# Failure State
# ------------------------------
class FailureState(BaseModel):
    brake_failure: bool = False
    sensor_failure: bool = False
    engine_overheating: bool = False
    low_oil: bool = False
    battery_issue: bool = False


# ------------------------------
# History (for memory)
# ------------------------------
class HistoryItem(BaseModel):
    state_summary: Dict[str, float]
    action_taken: Optional[str] = None


# ------------------------------
# Observation (WHAT AGENT SEES)
# ------------------------------
class Observation(BaseModel):
    speed: float = Field(..., ge=0, le=200)
    engine_temp: float = Field(..., ge=0, le=150)
    distance_to_obstacle: float = Field(..., ge=0, le=200)

    road_condition: str  # dry / wet / rain

    oil_level: float = Field(..., ge=0, le=100)
    battery_health: float = Field(..., ge=0, le=100)

    failures: FailureState
    history: List[HistoryItem]


# ------------------------------
# Action (WHAT AGENT DOES) ✅ FIXED
# ------------------------------
class Action(BaseModel):
    action_type: str  # brake / accelerate / turn_left / turn_right / continue / stop / request_service
    value: float = Field(..., ge=0, le=1)
    reason: str


# ------------------------------
# Reward (STEP LEVEL ONLY)
# ------------------------------
class Reward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)


# ------------------------------
# Metrics (FOR GRADER - FINAL EVAL)
# ------------------------------
class Metrics(BaseModel):
    safety_score: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    diagnosis_score: float = Field(..., ge=0.0, le=1.0)
    sequence_score: float = Field(..., ge=0.0, le=1.0)


# ------------------------------
# Step Result (OPENENV STYLE)
# ------------------------------
class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
    metrics: Metrics


# ------------------------------
# Episode State (INTERNAL USE)
# ------------------------------
class EpisodeState(BaseModel):
    step_count: int = 0
    max_steps: int = 20

    is_collision: bool = False
    is_engine_failure: bool = False
    is_safe_stop: bool = False

    def check_done(self) -> bool:
        if self.is_collision:
            return True
        if self.is_engine_failure:
            return True
        if self.is_safe_stop:
            return True
        if self.step_count >= self.max_steps:
            return True
        return False