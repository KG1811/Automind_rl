from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class FailureState(BaseModel):
    brake_failure: bool = False
    sensor_failure: bool = False
    engine_overheating: bool = False
    low_oil: bool = False
    battery_issue: bool = False


class HistoryItem(BaseModel):
    state_summary: Dict[str, float]
    action_taken: Optional[str] = None


class VehicleSignals(BaseModel):
    speed: float = Field(0.0, ge=0, le=220)
    rpm: float = Field(0.0, ge=0, le=8000)
    throttle: float = Field(0.0, ge=0, le=100)
    brake_pedal: float = Field(0.0, ge=0, le=100)
    gear: int = Field(0, ge=0, le=6)
    engine_load: float = Field(0.0, ge=0, le=100)
    transmission_load: float = Field(0.0, ge=0, le=100)
    fuel_rate: float = Field(0.0, ge=0, le=40)
    acceleration: float = Field(0.0, ge=-12, le=12)
    coolant_temp: float = Field(0.0, ge=0, le=150)
    oil_temp: float = Field(0.0, ge=0, le=170)
    oil_pressure: float = Field(0.0, ge=0, le=800)
    oil_level: float = Field(0.0, ge=0, le=100)
    battery_health: float = Field(0.0, ge=0, le=100)
    battery_voltage: float = Field(0.0, ge=0, le=18)
    fuel_level: float = Field(0.0, ge=0, le=100)
    distance_to_obstacle: float = Field(0.0, ge=0, le=300)
    drive_mode: str = "idle"
    road_condition: str = "dry"
    latitude: float = Field(0.0, ge=-90, le=90)
    longitude: float = Field(0.0, ge=-180, le=180)
    heading: float = Field(0.0, ge=0, le=360)
    odometer_km: float = Field(0.0, ge=0)
    ignition_on: bool = True
    charging_active: bool = False


class VehicleEvents(BaseModel):
    parked: bool = False
    trip_active: bool = False
    mil_status: bool = False
    dtc_count: int = Field(0, ge=0)
    dtc_codes: List[str] = Field(default_factory=list)
    overspeed_event: bool = False
    harsh_brake_event: bool = False
    low_battery_event: bool = False
    charging_fault: bool = False
    crash_event: bool = False
    battery_disconnect_event: bool = False
    engine_overheat_warning: bool = False
    low_oil_warning: bool = False
    brake_system_warning: bool = False
    sensor_fault_event: bool = False


class Observation(BaseModel):
    speed: float = Field(..., ge=0, le=220)
    rpm: float = Field(..., ge=0, le=8000)
    throttle: float = Field(..., ge=0, le=100)
    gear: int = Field(..., ge=0, le=6)
    engine_load: float = Field(..., ge=0, le=100)
    transmission_load: float = Field(..., ge=0, le=100)
    fuel_rate: float = Field(..., ge=0, le=40)
    acceleration: float = Field(..., ge=-12, le=12)

    engine_temp: float = Field(..., ge=0, le=150)
    distance_to_obstacle: float = Field(..., ge=0, le=300)
    road_condition: str
    drive_mode: str

    oil_level: float = Field(..., ge=0, le=100)
    battery_health: float = Field(..., ge=0, le=100)

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    heading: float = Field(..., ge=0, le=360)

    failures: FailureState
    history: List[HistoryItem]
    vehicle_signals: VehicleSignals = Field(default_factory=VehicleSignals)
    vehicle_events: VehicleEvents = Field(default_factory=VehicleEvents)


class TelemetryState(BaseModel):
    speed: float = Field(..., ge=0, le=220)
    rpm: float = Field(..., ge=0, le=8000)
    throttle: float = Field(..., ge=0, le=100)
    gear: int = Field(..., ge=0, le=6)
    engine_load: float = Field(..., ge=0, le=100)
    transmission_load: float = Field(..., ge=0, le=100)
    fuel_rate: float = Field(..., ge=0, le=40)
    acceleration: float = Field(..., ge=-12, le=12)

    engine_temp: float = Field(..., ge=0, le=150)
    distance_to_obstacle: float = Field(..., ge=0, le=300)
    road_condition: str
    drive_mode: str

    oil_level: float = Field(..., ge=0, le=100)
    battery_health: float = Field(..., ge=0, le=100)

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    heading: float = Field(..., ge=0, le=360)

    failures: FailureState
    vehicle_signals: VehicleSignals = Field(default_factory=VehicleSignals)
    vehicle_events: VehicleEvents = Field(default_factory=VehicleEvents)


class Action(BaseModel):
    action_type: str
    value: float = Field(..., ge=0, le=1)
    reason: str


class Metrics(BaseModel):
    safety_score: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    diagnosis_score: float = Field(..., ge=0.0, le=1.0)
    sequence_score: float = Field(..., ge=0.0, le=1.0)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
    metrics: Metrics


class EpisodeState(BaseModel):
    step_count: int = 0
    max_steps: int = 20
    is_collision: bool = False
    is_engine_failure: bool = False
    is_safe_stop: bool = False

    def check_done(self) -> bool:
        return (
            self.is_collision
            or self.is_engine_failure
            or self.is_safe_stop
            or self.step_count >= self.max_steps
        )
