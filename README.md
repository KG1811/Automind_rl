# AutoMind OpenEnv

AutoMind OpenEnv is a real-world automotive decision environment for agent evaluation.

## What it simulates
- ECU telemetry: speed, RPM, throttle, engine load, fuel rate, engine temperature
- TCU telemetry: gear, transmission load, drive mode
- Vehicle context: road condition, obstacle distance, acceleration
- Health and alerting: overheating, low oil, battery issue, collision risk
- GPS movement and nearest-service recommendation
- Human override behavior
- Continuous live background telemetry updates

## Required APIs
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /schema`

## Tasks
1. `fault_diagnosis`
2. `driving_decision`
3. `autonomous_control`

## Reward
Dense step reward based on:
- safety
- efficiency
- action quality
- sequence quality

## Metrics
- safety_score
- efficiency_score
- diagnosis_score
- sequence_score

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000