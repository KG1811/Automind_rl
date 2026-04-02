# AutoMind OpenEnv Environment

## Overview
AutoMind OpenEnv is a real-world automotive decision-making environment designed for training and evaluating AI agents.

The agent must:
- Diagnose vehicle faults
- Make safe driving decisions
- Handle failures
- Decide whether to continue or stop
- Recommend service actions

---

## Observation Space
The agent observes:
- Speed
- Engine temperature
- Distance to obstacle
- Road condition (dry / wet / rain)
- Oil level
- Battery health
- Failure states (brake failure, overheating, etc.)
- Action history (memory)

---

## Action Space
The agent can perform:
- `brake`
- `accelerate`
- `turn`
- `continue`
- `stop`
- `request_service`

---

## Tasks

### 1. Fault Diagnosis (Easy)
- Identify faults like overheating, low oil, battery issues

### 2. Driving Decision (Medium)
- Choose safe driving actions based on environment state

### 3. Autonomous Control (Hard)
- Full pipeline:
  - Diagnose faults
  - Handle failures
  - Avoid collisions
  - Adapt to human override
  - Decide service vs continue

---

## Episode Design
An episode terminates when:
- Collision occurs
- Engine failure occurs
- Vehicle safely stops
- Maximum steps reached

---

## Reward vs Grader

- **Reward** → Step-wise feedback for agent learning  
- **Grader** → Final evaluation score (0.0–1.0)

This separation ensures:
- Stable RL training
- Fair evaluation

---

## API Endpoints

### POST `/reset`
Initialize environment

### POST `/step`
Send action:
```json
{
  "action_type": "brake",
  "value": 0.7
}