# AutoMind OpenEnv

AutoMind OpenEnv is a highly realistic automotive simulator built for the validation and benchmarking of AI Agents. The environment natively simulates ECU (Engine Control Unit) physics, vehicle dynamics, predictive failures, sensor noise, and environmental traffic pressure to create a deterministic, API-first testing ground for autonomous driving and diagnostic policies.

## Table of Contents
- [Tasks & Grading](#tasks--grading)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Installation & Local Setup](#installation--local-setup)
- [Deploying to HuggingFace / Docker](#deploying-to-huggingface--docker)
- [Baseline Inference & Scores](#baseline-inference--scores)

---

## Tasks & Grading

AutoMind evaluates agents across three increasing difficulty tiers:

1. **`fault_diagnosis`**: Given raw ECU data, correctly diagnose ongoing engine or battery failures (Overheating, Low Oil, Battery Issue). Target action: `{"action_type": "diagnose", "reason": "<fault>"}`. (Difficulty: Easy).
2. **`driving_decision`**: Evaluate live conditions and output the safest operational driving decision (`brake`, `accelerate`, etc.) based on speed and RPM. (Difficulty: Medium).
3. **`autonomous_control`**: Full 20-step episode loop. The agent must sustain optimal driving, handle live human overrides safely, keep health high, and call for nearest-service interventions appropriately. (Difficulty: Hard).

All grader endpoints output a `0.0` to `1.0` normalized float score representing the success of the agent.

---

## Action Space

The server receives JSON matching the following schema:

```json
{
  "action_type": "string (enum)",
  "value": "float [0.0, 1.0]",
  "reason": "string"
}
```

**Allowed Actions:**
- `diagnose` (for task 1)
- `brake`
- `accelerate`
- `turn_left`
- `turn_right`
- `continue`
- `stop`
- `request_service`

---

## Observation Space

The `Observation` received back maps strictly to physical telemetry and proxy metrics. None of the variables hold explicitly "solved" game state (e.g. `distance_to_obstacle` is purely hidden ground-truth, forcing the agent to infer collision risk from speed/throttle behavior).

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `speed` | Float | 0 - 220 km/h | Absolute velocity. |
| `rpm` | Float | 0 - 8000 | Current engine RPM. |
| `throttle` | Float | 0 - 100% | Pedal depress percentage. |
| `gear` | Int | 0 - 6 | Current transmission gear. |
| `engine_load`| Float | 0 - 100% | System load calculated by logic. |
| `fuel_rate` | Float | 0 - 40 L/hr| Current consumption. |
| `acceleration`| Float| -12 - +12 | Derivative of speed over time. |
| `engine_temp` | Float | 0 - 150 C | Live temperature tracking. |
| `oil_level` | Float | 0 - 100% | Remaining healthy oil %. |
| `battery_health`| Float | 0 - 100% | High Voltage Battery %. |
| `failures` | Object | boolean flags | Detected catastrophic faults. |
| `history` | Array | past 8 steps | Short-term memory stream. |

---

## Installation & Local Setup

### Python Virtual Environment

```bash
# Clone the repo and enter directory
git clone <repo>
cd automind-openenv

# Install requirements
pip install -r requirements.txt

# Launch FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server natively exposes standard OpenEnv hooks: `/reset`, `/step`, `/state`, `/schema`, `/tasks`.

---

## Deploying to HuggingFace / Docker

You can containerize this environment using the included multi-stage Docker deployment, which perfectly wraps the FastAPI interface for HuggingFace Spaces.

```dockerfile
# Build image
docker build -t automind-env .

# Run container (Exposes on 8080 or port defined in Dockerfile)
docker run -p 8080:8000 automind-env
```

**HuggingFace Deployment Instructions:**
1. Create a new "Docker" Space on HuggingFace.
2. Link the repository GitHub branch.
3. HuggingFace will natively run the `Dockerfile` and expose the API via the standard Space URL (e.g., `https://your-user-automind.hf.space`).

---

## Baseline Inference & Scores

You can execute a full loop benchmark using the bundled baseline evaluator. Ensure your environment variables contain the required tokens for AI inference.

```bash
export API_BASE_URL="http://127.0.0.1:8000"
export HF_TOKEN="<your_openai_or_hf_token>"

python inference.py
```

### Reproducible Baseline Results (Mock Target Example)
*Score averages run deterministically on seeded physics iterations.*

```text
==============================
      BASELINE RESULTS
==============================

[FAULT_DIAGNOSIS]
  Easy: 1.000
  Medium: 1.000
  Hard: 1.000
  --> Average: 1.000

[DRIVING_DECISION]
  Easy: 0.810
  Medium: 0.740
  Hard: 0.615
  --> Average: 0.721

[AUTONOMOUS_CONTROL]
  Easy: 0.892
  Medium: 0.814
  Hard: 0.710
  --> Average: 0.805
```