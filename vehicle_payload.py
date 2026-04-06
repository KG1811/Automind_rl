from __future__ import annotations

import random

from models import FailureState, VehicleEvents, VehicleSignals


FUEL_TANK_CAPACITY_L = 50.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_brake_pedal(action_type: str, action_value: float) -> float:
    if action_type == "stop":
        return 100.0
    if action_type == "brake":
        return clamp(15.0 + (85.0 * action_value), 0.0, 100.0)
    if action_type == "request_service":
        return 18.0
    if action_type in {"turn_left", "turn_right"}:
        return 8.0
    return 0.0


def compute_ignition_on(speed: float, throttle: float, action_type: str) -> bool:
    if action_type in {"stop", "request_service"} and speed < 0.5 and throttle < 1.0:
        return False
    return True


def compute_charging_active(
    ignition_on: bool,
    battery_issue_active: bool,
    throttle: float,
    speed: float,
) -> bool:
    if not ignition_on or battery_issue_active:
        return False
    return speed > 1.0 or throttle > 3.0


def compute_oil_temp(engine_temp: float, rpm: float, action_type: str, rng: random.Random) -> float:
    target = engine_temp + 4.0 + (rpm / 5200.0) * 14.0
    if action_type in {"brake", "stop"}:
        target -= 4.5
    return clamp(target + rng.uniform(-0.6, 0.6), 35.0, 170.0)


def compute_oil_pressure(
    rpm: float,
    oil_level: float,
    oil_temp: float,
    low_oil_active: bool,
) -> float:
    base = 110.0 + rpm * 0.065
    temp_penalty = max(0.0, oil_temp - 110.0) * 1.4
    oil_penalty = max(0.0, 25.0 - oil_level) * 3.0
    if low_oil_active:
        oil_penalty += 45.0
    return clamp(base - temp_penalty - oil_penalty, 35.0, 780.0)


def update_fuel_level(
    current_fuel_level: float,
    fuel_rate: float,
    dt_seconds: float,
    parked: bool,
) -> float:
    consumed_liters = fuel_rate * (dt_seconds / 3600.0)
    consumed_pct = (consumed_liters / FUEL_TANK_CAPACITY_L) * 100.0
    if parked:
        consumed_pct *= 0.35
    return clamp(current_fuel_level - consumed_pct, 0.0, 100.0)


def update_odometer(current_odometer_km: float, speed: float, dt_seconds: float) -> float:
    return max(0.0, current_odometer_km + speed * (dt_seconds / 3600.0))


def compute_battery_voltage(
    battery_health: float,
    ignition_on: bool,
    charging_active: bool,
    battery_issue_active: bool,
    throttle: float,
    speed: float,
    rng: random.Random,
) -> float:
    if not ignition_on:
        base_voltage = 12.55
    elif charging_active:
        base_voltage = 13.7 + min(0.7, (throttle / 100.0) * 0.4 + (speed / 160.0) * 0.3)
    else:
        base_voltage = 12.2

    health_penalty = (100.0 - battery_health) * 0.006
    if battery_issue_active:
        health_penalty += 0.55

    return clamp(base_voltage - health_penalty + rng.uniform(-0.08, 0.08), 9.8, 14.8)


def build_dtc_codes(
    failures: FailureState,
    engine_temp: float,
    oil_level: float,
    battery_voltage: float,
    oil_pressure: float,
    is_collision: bool,
) -> list[str]:
    codes: list[str] = []

    if failures.engine_overheating or engine_temp >= 110.0:
        codes.append("P0217")
    if failures.low_oil or oil_level <= 25.0 or oil_pressure < 90.0:
        codes.append("P0522")
    if failures.battery_issue or battery_voltage < 11.8:
        codes.append("P0562")
    if failures.brake_failure:
        codes.append("C0040")
    if failures.sensor_failure:
        codes.append("U0121")
    if is_collision:
        codes.append("B0001")

    return codes


def build_vehicle_signals(
    *,
    speed: float,
    rpm: float,
    throttle: float,
    action_type: str,
    action_value: float,
    gear: int,
    engine_load: float,
    transmission_load: float,
    fuel_rate: float,
    acceleration: float,
    engine_temp: float,
    oil_level: float,
    battery_health: float,
    distance_to_obstacle: float,
    road_condition: str,
    drive_mode: str,
    latitude: float,
    longitude: float,
    heading: float,
    previous_fuel_level: float,
    previous_odometer_km: float,
    battery_issue_active: bool,
    low_oil_active: bool,
    dt_seconds: float,
    rng: random.Random,
) -> VehicleSignals:
    brake_pedal = compute_brake_pedal(action_type=action_type, action_value=action_value)
    ignition_on = compute_ignition_on(speed=speed, throttle=throttle, action_type=action_type)
    charging_active = compute_charging_active(
        ignition_on=ignition_on,
        battery_issue_active=battery_issue_active,
        throttle=throttle,
        speed=speed,
    )
    oil_temp = compute_oil_temp(
        engine_temp=engine_temp,
        rpm=rpm,
        action_type=action_type,
        rng=rng,
    )
    oil_pressure = compute_oil_pressure(
        rpm=rpm,
        oil_level=oil_level,
        oil_temp=oil_temp,
        low_oil_active=low_oil_active,
    )
    parked = speed < 1.0 and not ignition_on
    fuel_level = update_fuel_level(
        current_fuel_level=previous_fuel_level,
        fuel_rate=fuel_rate,
        dt_seconds=dt_seconds,
        parked=parked,
    )
    odometer_km = update_odometer(
        current_odometer_km=previous_odometer_km,
        speed=speed,
        dt_seconds=dt_seconds,
    )
    battery_voltage = compute_battery_voltage(
        battery_health=battery_health,
        ignition_on=ignition_on,
        charging_active=charging_active,
        battery_issue_active=battery_issue_active,
        throttle=throttle,
        speed=speed,
        rng=rng,
    )

    return VehicleSignals(
        speed=round(speed, 2),
        rpm=round(rpm, 2),
        throttle=round(throttle, 2),
        brake_pedal=round(brake_pedal, 2),
        gear=gear,
        engine_load=round(engine_load, 2),
        transmission_load=round(transmission_load, 2),
        fuel_rate=round(fuel_rate, 2),
        acceleration=round(acceleration, 2),
        coolant_temp=round(engine_temp, 2),
        oil_temp=round(oil_temp, 2),
        oil_pressure=round(oil_pressure, 2),
        oil_level=round(oil_level, 2),
        battery_health=round(battery_health, 2),
        battery_voltage=round(battery_voltage, 2),
        fuel_level=round(fuel_level, 2),
        distance_to_obstacle=round(distance_to_obstacle, 2),
        drive_mode=drive_mode,
        road_condition=road_condition,
        latitude=round(latitude, 6),
        longitude=round(longitude, 6),
        heading=round(heading, 2),
        odometer_km=round(odometer_km, 3),
        ignition_on=ignition_on,
        charging_active=charging_active,
    )


def build_vehicle_events(
    *,
    signals: VehicleSignals,
    failures: FailureState,
    is_collision: bool,
) -> VehicleEvents:
    dtc_codes = build_dtc_codes(
        failures=failures,
        engine_temp=signals.coolant_temp,
        oil_level=signals.oil_level,
        battery_voltage=signals.battery_voltage,
        oil_pressure=signals.oil_pressure,
        is_collision=is_collision,
    )

    parked = signals.speed < 1.0 and not signals.ignition_on
    trip_active = signals.ignition_on and (signals.speed > 1.0 or signals.fuel_rate > 0.8)
    overspeed_event = signals.speed >= 100.0
    harsh_brake_event = signals.acceleration <= -5.5 or signals.brake_pedal >= 70.0
    low_battery_event = signals.battery_voltage < 11.8 or signals.battery_health < 25.0
    charging_fault = signals.ignition_on and not signals.charging_active and signals.battery_voltage < 12.0
    battery_disconnect_event = not signals.ignition_on and signals.battery_voltage < 10.8

    return VehicleEvents(
        parked=parked,
        trip_active=trip_active,
        mil_status=len(dtc_codes) > 0,
        dtc_count=len(dtc_codes),
        dtc_codes=dtc_codes,
        overspeed_event=overspeed_event,
        harsh_brake_event=harsh_brake_event,
        low_battery_event=low_battery_event,
        charging_fault=charging_fault,
        crash_event=is_collision,
        battery_disconnect_event=battery_disconnect_event,
        engine_overheat_warning=failures.engine_overheating or signals.coolant_temp >= 110.0,
        low_oil_warning=failures.low_oil or signals.oil_level <= 25.0 or signals.oil_pressure < 90.0,
        brake_system_warning=failures.brake_failure,
        sensor_fault_event=failures.sensor_failure,
    )
