# Structured Cognitive Loop (SCL) with Context-Aware Human-in-the-Loop

This repository contains the experimental implementation for two papers:

1. **"Structured Cognitive Loop: Bridging Symbolic Control and Neural Reasoning in LLM Agents"**
2. **"Beyond Static Interrupts: Context-Aware Human-in-the-Loop as a Cognitive Process for Trustworthy LLM Agents"**

by Myung Ho Kim (JEI University)

## Overview

This implementation demonstrates the **R-CCAM architecture** with integrated **Context-Aware HITL**, treating human intervention as a first-class cognitive event rather than a static safety interrupt.

### Key Features

| Feature | Description |
|---------|-------------|
| **Cognitive State Freezing** | Pause execution and preserve complete cognitive state |
| **Cognitive State Thawing** | Resume from exact frozen state without losing context |
| **Virtual Rejection Cycles** | Re-enter cognition with human feedback after rejection |
| **Action-Centric Intervention** | Intervene at action boundaries, not arbitrary points |
| **Glassbox Trace** | Complete audit trail including all human decisions |

## Architecture

```
                                                        ┌───────────────────┐
                                                        │    Metaprompt     │
                                                        │ (Soft Symbolic    │
                                                        │  Control Layer)   │
                                                        └─────────┬─────────┘
                                                                  │
                                                                  ▼
                         ┌────────────────────────────────────────────────────┐
                         │              Retrieval (Once at Start)             │
                         └────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
        ┌──────┐         ┌────────────────────────────────────────────────────┐
        │      │         │              Cognition (With Memory)               │◄───────┐
        │      │         └────────────────────────┬───────────────────────────┘        │
        │      │                                  │                                    │
        │ Loop │                                  ▼                                    │
        │Until │         ┌────────────────────────────────────────────────────┐        │
        │ Done │         │                    Control                         │        │
        │      │         └──────────┬─────────────────────────┬───────────────┘        │
        │      │                    │                         │                        │
        │      │                    │ Safe = True             │ HITL Check:            │
        │      │                    │                         │ State Freeze           │
        │      │                    │                         ▼                        │
        │      │                    │              ┌─────────────────────┐             │
        │      │                    │              │   Human Decision    │             │
        │      │                    │              └──┬─────────┬─────┬──┘             │
        │      │                    │                 │         │     │                │
        │      │                    │        Approve  │      Modify   │ Reject         │
        │      │                    │     State Thaw  │    State Thaw │ State Thaw     │
        │      │                    │                 │         │     │                │
        │      │                    ▼                 ▼         │     │                │
        │      │         ┌───────────────────────────────┐      │     │                │
        │      │         │           Action              │◄─────┘     │                │
        │      │         └──────────────┬────────────────┘            │                │
        │      │                        │                             │                │
        │      │                        │                             ▼                │
        │      │                        │              ┌─────────────────────────┐     │
        │      │                        │              │   Virtual Rejection     │     │
        │      │                        │              │   Cycle (with Feedback) │     │
        │      │                        │              └─────────────┬───────────┘     │
        │      │                        │                            │                 │
        │      │                        ▼                            │                 │
        │      │         ┌────────────────────────────────────────────────────┐        │
        │      │         │                  Memory Update                     │◄───────┘
        │      │         └────────────────────────┬───────────────────────────┘
        │      │                                  │
        │      │                                  ▼
        │      │         ┌────────────────────────────────────────────────────┐
        │      │         │              Done? (Task Complete)                 │
        └──────┘         └────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
                         ┌────────────────────────────────────────────────────┐
                         │            Audit Log (Glassbox Trace)              │
                         └────────────────────────────────────────────────────┘
```

## Files

```
scl_core_with_hitl.py   - Core R-CCAM loop with HITL integration
hitl_module.py          - HITL module (freezing, thawing, policies)
mock_tools.py           - Tool registry and mock implementations
mock_cognition.py       - Mock LLM cognition engine
run_experiment_hitl.py  - Main experiment runner with HITL
README_HITL.md          - This documentation
```

## Quick Start

### Basic Execution (Auto-Approve Mode)

```bash
python run_experiment_hitl.py
```

### Interactive Mode (Manual Decisions)

```bash
python run_experiment_hitl.py --interactive
```

### Run Demonstrations

```bash
python run_experiment_hitl.py --demo
```

### Disable HITL

```bash
python run_experiment_hitl.py --disabled
```

## HITL Concepts Implemented

### 1. Cognitive State Freezing and Thawing

When human intervention is required, the system:

1. **Freezes** the complete cognitive state (loop counter, memory, evidence, context)
2. Requests human decision (approve/reject/modify)
3. **Thaws** the state to resume execution exactly where it paused

```python
# Freeze state
frozen = hitl_manager.freeze_state(
    loop_counter=3,
    cognition_output=cognition_output,
    memory_state=memory.get_snapshot(),
    evidence_cache=memory.get_evidence_snapshot(),
    context=context,
    metaprompt_state=metaprompt.get_state(),
    intervention_level=InterventionLevel.APPROVE,
    intervention_reason="High-risk tool: send_email"
)

# Later: Thaw and resume
thawed = hitl_manager.thaw_state(frozen.freeze_id)
```

### 2. Virtual Rejection Cycles

When human rejects an action, instead of failing:

1. Create a **virtual rejection cycle**
2. **Skip Action entirely** — go directly to Memory Update
3. Store rejection feedback in Memory
4. Loop back to Cognition with awareness of rejection

```python
# After rejection
rejection_cycle = hitl_manager.create_virtual_rejection_cycle(
    frozen_state,
    "Wrong email address - use correct one"
)

# Context for next cognition cycle
context.update({
    "human_rejected": True,
    "rejection_reason": "Wrong email address",
    "retry_guidance": "Consider alternative approaches"
})

# Note: Action is NOT executed - control goes directly to Memory Update
```

### 3. Action-Centric Intervention

HITL checks happen at the **action boundary** (after Control, before Action):

- Cognition proposes action
- Control validates against rules
- **HITL Check**: Evaluate if human intervention needed
  - **Approve/Modify** → Action → Memory Update
  - **Reject** → Virtual Rejection Cycle → Memory Update (bypass Action)

### 4. HITL Policy Configuration

```python
policy = HITLPolicy()
policy.policies.update({
    "high_risk_tools": ["send_email", "cancel_trip", "delete_data"],
    "always_confirm_tools": ["generate_image"],
    "confirm_on_final_action": True,
    "confirm_on_confidence_below": 0.8,
    "confirm_after_n_loops": 10,
})
```

### 5. Intervention Levels

| Level | Behavior |
|-------|----------|
| `NONE` | No intervention needed |
| `NOTIFY` | Inform human, proceed automatically |
| `CONFIRM` | Require explicit confirmation |
| `APPROVE` | Require approval (approve/reject/modify) |
| `BLOCK` | Block until human decision |

## Glassbox Trace

All events are logged for complete auditability:

```json
{
  "hitl_events": [
    {
      "trace_id": "HITL-0001",
      "timestamp": "2024-01-15T10:30:00Z",
      "event_type": "state_frozen",
      "freeze_id": "FREEZE-0001",
      "pending_action": {"tool_name": "send_email", ...},
      "actor": "system"
    },
    {
      "trace_id": "HITL-0002",
      "timestamp": "2024-01-15T10:30:15Z",
      "event_type": "approved",
      "freeze_id": "FREEZE-0001",
      "human_decision": "approve",
      "decision_rationale": "Confirmed destination is correct",
      "actor": "human"
    }
  ],
  "statistics": {
    "total_interventions": 4,
    "approvals": 1,
    "rejections": 0,
    "modifications": 0
  }
}
```

## Example Output

```
############################################################
# STRUCTURED COGNITIVE LOOP (SCL) WITH HITL
# Mode: auto
############################################################

[RETRIEVAL] Initializing task...

[COGNITION] Loop 1
Reasoning: Need weather data for San Francisco...
Proposed Action: {'tool_name': 'get_weather', ...}

[CONTROL] Validating proposed action...
✓ PASS: PASS

[ACTION] Executing validated action...
Executed: get_weather
Result: {'city': 'San Francisco', 'temperature_f': 55, ...}

... (loops 2-3 for Miami and Atlanta) ...

[COGNITION] Loop 4
Reasoning: Two cities above base temperature. Send email notification.
Proposed Action: {'tool_name': 'send_email', ...}

[CONTROL] Validating proposed action...
✓ PASS: PASS

[HITL] Intervention required: approve
       Reason: High-risk tool: send_email

❄️  [HITL] State frozen: FREEZE-0001
    Reason: High-risk tool: send_email
    Level: approve
    Pending: send_email

🤖 [AUTO-HITL] Auto-approving: send_email

👤 [HITL] Human decision: APPROVE
    Rationale: Auto-approved for testing

🔥 [HITL] State thawed: FREEZE-0001

[ACTION] Executing validated action...
📧 EMAIL SENT
To: test-scl@test.com
Subject: Travel Plan Confirmed: Atlanta

[COMPLETION] Task finished in 4 loops

================================================================================
EXPERIMENT SUMMARY STATISTICS
================================================================================

📊 Performance Metrics:
   • Total CCAM loops: 4
   • Policy violations: 0
   • Success rate: 100.0%

👤 HITL Metrics:
   • Total interventions: 4
   • Approvals: 1
   • Rejections: 0

🔧 Architecture Validation:
   ✓ Modular decomposition maintained
   ✓ Soft Symbolic Control enforced
   ✓ Context-Aware HITL integrated
   ✓ Cognitive state freezing/thawing operational
   ✓ Glassbox Trace generated
```

## Production Integration

### Replace Mock Components

```python
# Real LLM integration
from openai import OpenAI

class GPT5CognitionEngine:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def __call__(self, prompt: str, context: Dict) -> Dict:
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

### Custom HITL Handlers (for Web UI)

```python
class WebUIHITLHandler:
    def __init__(self, websocket):
        self.ws = websocket
    
    async def handle_intervention(self, request: Dict) -> Dict:
        # Send to frontend
        await self.ws.send(json.dumps(request))
        
        # Wait for user response
        response = await self.ws.recv()
        return json.loads(response)
```

## Comparison with Static HITL

| Aspect | Static HITL | Context-Aware HITL (This Work) |
|--------|-------------|-------------------------------|
| When | Fixed points | Action boundaries |
| State | Lost on pause | Frozen/Thawed |
| Rejection | Fails task | Virtual rejection cycle |
| Logging | Minimal | Glassbox Trace |
| Resumption | Restart | Exact state |
| Context | None | Full cognitive context |

## Citation

```bibtex
@techreport{kim2025hitl,
  title={Beyond Static Interrupts: Context-Aware Human-in-the-Loop 
         as a Cognitive Process for Trustworthy LLM Agents},
  author={Kim, Myung Ho},
  institution={JEI University},
  year={2025}
}
```

## Contact

- **Author**: Myung Ho Kim
- **Email**: enkiluv@gmail.com
- **ORCID**: 0009-0001-3709-7622

## License

MIT License - See LICENSE file for details
