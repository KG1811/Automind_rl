# ==============================
# AutoMind OpenEnv - Noise Engine
# ==============================

from __future__ import annotations

import random


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def add_sensor_noise(
    rng: random.Random,
    value: float,
    std_dev: float,
    low: float,
    high: float,
) -> float:
    noisy = value + rng.gauss(0.0, std_dev)
    return clamp(noisy, low, high)


def maybe_corrupt_distance(
    rng: random.Random,
    value: float,
    sensor_failure: bool,
) -> float:
    """
    If sensor failure is active, distance reading becomes less reliable.
    """
    if sensor_failure:
        return clamp(value + rng.gauss(0.0, 6.0), 0.0, 200.0)
    return clamp(value + rng.gauss(0.0, 1.5), 0.0, 200.0)