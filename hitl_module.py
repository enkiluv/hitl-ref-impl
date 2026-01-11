"""
Human-in-the-Loop (HITL) Module for Structured Cognitive Loop (SCL)

Based on the paper: "Beyond Static Interrupts: Context-Aware Human-in-the-Loop 
as a Cognitive Process for Trustworthy LLM Agents"

Key Concepts Implemented:
1. Cognitive State Freezing and Thawing - Pause/resume without losing context
2. Virtual Rejection Cycles - Handle human rejections as cognitive events
3. Trace-Grounded Resumption - Resume from exact cognitive state
4. Action-Centric Intervention - Intervene at action boundaries
5. Human Judgment as Normative Cognitive Event - Log all human decisions
"""

import json
import copy
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum


class HITLEventType(Enum):
    """Types of HITL events in the cognitive loop"""
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    STATE_FROZEN = "state_frozen"
    STATE_THAWED = "state_thawed"
    TIMEOUT = "timeout"
    DELEGATED = "delegated"


class InterventionLevel(Enum):
    """Levels of human intervention required"""
    NONE = "none"                    # No intervention needed
    NOTIFY = "notify"                # Inform human, but proceed
    CONFIRM = "confirm"              # Require explicit confirmation
    APPROVE = "approve"              # Require approval with possible modification
    BLOCK = "block"                  # Block until human decision


@dataclass
class FrozenCognitiveState:
    """
    Frozen cognitive state for pause/resume functionality
    Enables trace-grounded resumption without loss of epistemic continuity
    """
    freeze_id: str
    freeze_timestamp: str
    loop_counter: int
    pending_action: Dict[str, Any]
    cognition_output: Dict[str, Any]
    memory_snapshot: Dict[str, Any]
    evidence_cache: Dict[str, Any]
    context: Dict[str, Any]
    metaprompt_state: Dict[str, Any]
    intervention_reason: str
    intervention_level: InterventionLevel
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['intervention_level'] = self.intervention_level.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrozenCognitiveState':
        data['intervention_level'] = InterventionLevel(data['intervention_level'])
        return cls(**data)


@dataclass
class HITLTrace:
    """
    Record of HITL event for Glassbox Trace
    All human interventions are recorded as verifiable cognitive transitions
    """
    trace_id: str
    timestamp: str
    event_type: HITLEventType
    freeze_id: Optional[str]
    pending_action: Dict[str, Any]
    human_decision: Optional[str]
    human_feedback: Optional[str]
    modified_action: Optional[Dict[str, Any]]
    decision_rationale: Optional[str]
    actor: str  # "human" or "system"
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['event_type'] = self.event_type.value
        return result


class HITLPolicy:
    """
    Policy configuration for when to require human intervention
    Implements Action-Centric Intervention
    """
    
    def __init__(self):
        # Default policies - can be customized
        self.policies = {
            # Tool-based policies
            "high_risk_tools": ["send_email", "cancel_trip", "make_payment", "delete_data"],
            "always_confirm_tools": ["generate_image"],
            
            # Condition-based policies
            "confirm_on_final_action": True,
            "confirm_on_external_action": True,
            "confirm_on_cost_threshold": 100.0,
            "confirm_on_confidence_below": 0.7,
            
            # Loop-based policies
            "confirm_after_n_loops": 10,
            "confirm_on_loop_retry": True,
            
            # Evidence-based policies
            "confirm_on_missing_evidence": True,
            "confirm_on_conflicting_evidence": True,
        }
        
        # Custom intervention handlers
        self.custom_handlers: Dict[str, Callable] = {}
    
    def register_handler(self, tool_name: str, handler: Callable):
        """Register custom intervention handler for specific tool"""
        self.custom_handlers[tool_name] = handler
    
    def evaluate(
        self, 
        cognition_output: Dict[str, Any],
        context: Dict[str, Any],
        loop_counter: int
    ) -> tuple[InterventionLevel, str]:
        """
        Evaluate whether human intervention is required
        Returns (intervention_level, reason)
        """
        proposed_action = cognition_output.get("proposed_action", {})
        tool_name = proposed_action.get("tool_name", "")
        
        # Check high-risk tools
        if tool_name in self.policies["high_risk_tools"]:
            return InterventionLevel.APPROVE, f"High-risk tool: {tool_name}"
        
        # Check always-confirm tools
        if tool_name in self.policies["always_confirm_tools"]:
            return InterventionLevel.CONFIRM, f"Confirmation required for: {tool_name}"
        
        # Check final action
        if self.policies["confirm_on_final_action"] and cognition_output.get("is_final_action"):
            return InterventionLevel.CONFIRM, "Final action requires confirmation"
        
        # Check loop count
        if loop_counter >= self.policies["confirm_after_n_loops"]:
            return InterventionLevel.NOTIFY, f"Extended loop count: {loop_counter}"
        
        # Check confidence (if available)
        confidence = cognition_output.get("confidence", 1.0)
        if confidence < self.policies["confirm_on_confidence_below"]:
            return InterventionLevel.CONFIRM, f"Low confidence: {confidence:.2f}"
        
        # Check missing evidence
        if self.policies["confirm_on_missing_evidence"]:
            if not cognition_output.get("evidence_refs"):
                return InterventionLevel.NOTIFY, "Missing evidence citations"
        
        # Check custom handlers
        if tool_name in self.custom_handlers:
            return self.custom_handlers[tool_name](cognition_output, context)
        
        return InterventionLevel.NONE, ""


class HITLManager:
    """
    Human-in-the-Loop Manager
    Manages cognitive state freezing/thawing, intervention requests, and trace logging
    """
    
    def __init__(self, policy: Optional[HITLPolicy] = None):
        self.policy = policy or HITLPolicy()
        self.frozen_states: Dict[str, FrozenCognitiveState] = {}
        self.hitl_traces: List[HITLTrace] = []
        self.freeze_counter = 0
        self.trace_counter = 0
        
        # Callbacks for UI integration
        self.on_intervention_required: Optional[Callable] = None
        self.on_state_frozen: Optional[Callable] = None
        self.on_state_thawed: Optional[Callable] = None
    
    def _generate_freeze_id(self) -> str:
        self.freeze_counter += 1
        return f"FREEZE-{self.freeze_counter:04d}"
    
    def _generate_trace_id(self) -> str:
        self.trace_counter += 1
        return f"HITL-{self.trace_counter:04d}"
    
    def check_intervention(
        self,
        cognition_output: Dict[str, Any],
        context: Dict[str, Any],
        loop_counter: int
    ) -> tuple[InterventionLevel, str]:
        """Check if human intervention is required for the proposed action"""
        return self.policy.evaluate(cognition_output, context, loop_counter)
    
    def freeze_state(
        self,
        loop_counter: int,
        cognition_output: Dict[str, Any],
        memory_state: Dict[str, Any],
        evidence_cache: Dict[str, Any],
        context: Dict[str, Any],
        metaprompt_state: Dict[str, Any],
        intervention_level: InterventionLevel,
        intervention_reason: str
    ) -> FrozenCognitiveState:
        """
        Freeze the current cognitive state for later resumption
        Implements Cognitive State Freezing
        """
        freeze_id = self._generate_freeze_id()
        
        frozen = FrozenCognitiveState(
            freeze_id=freeze_id,
            freeze_timestamp=datetime.now().isoformat(),
            loop_counter=loop_counter,
            pending_action=cognition_output.get("proposed_action", {}),
            cognition_output=copy.deepcopy(cognition_output),
            memory_snapshot=copy.deepcopy(memory_state),
            evidence_cache=copy.deepcopy(evidence_cache),
            context=copy.deepcopy(context),
            metaprompt_state=copy.deepcopy(metaprompt_state),
            intervention_reason=intervention_reason,
            intervention_level=intervention_level
        )
        
        self.frozen_states[freeze_id] = frozen
        
        # Log freeze event
        self._log_trace(
            event_type=HITLEventType.STATE_FROZEN,
            freeze_id=freeze_id,
            pending_action=frozen.pending_action,
            actor="system"
        )
        
        # Callback for UI
        if self.on_state_frozen:
            self.on_state_frozen(frozen)
        
        print(f"\n❄️  [HITL] State frozen: {freeze_id}")
        print(f"    Reason: {intervention_reason}")
        print(f"    Level: {intervention_level.value}")
        print(f"    Pending: {frozen.pending_action.get('tool_name', 'N/A')}")
        
        return frozen
    
    def thaw_state(self, freeze_id: str) -> Optional[FrozenCognitiveState]:
        """
        Thaw a frozen cognitive state for resumption
        Implements Cognitive State Thawing
        """
        if freeze_id not in self.frozen_states:
            print(f"⚠️  [HITL] Frozen state not found: {freeze_id}")
            return None
        
        frozen = self.frozen_states[freeze_id]
        
        # Log thaw event
        self._log_trace(
            event_type=HITLEventType.STATE_THAWED,
            freeze_id=freeze_id,
            pending_action=frozen.pending_action,
            actor="system"
        )
        
        if self.on_state_thawed:
            self.on_state_thawed(frozen)
        
        print(f"\n🔥 [HITL] State thawed: {freeze_id}")
        
        return frozen
    
    def request_approval(
        self,
        frozen_state: FrozenCognitiveState
    ) -> Dict[str, Any]:
        """
        Request human approval for pending action
        Returns approval request details for UI
        """
        # Log approval request
        self._log_trace(
            event_type=HITLEventType.APPROVAL_REQUESTED,
            freeze_id=frozen_state.freeze_id,
            pending_action=frozen_state.pending_action,
            actor="system"
        )
        
        request = {
            "freeze_id": frozen_state.freeze_id,
            "timestamp": datetime.now().isoformat(),
            "intervention_level": frozen_state.intervention_level.value,
            "intervention_reason": frozen_state.intervention_reason,
            "pending_action": frozen_state.pending_action,
            "cognition_reasoning": frozen_state.cognition_output.get("reasoning", ""),
            "evidence_refs": frozen_state.cognition_output.get("evidence_refs", []),
            "context_summary": {
                "loop_counter": frozen_state.loop_counter,
                "task": frozen_state.context.get("task", "")[:200],
                "last_action": str(frozen_state.context.get("last_action_result", ""))[:200]
            },
            "options": ["approve", "reject", "modify"]
        }
        
        if self.on_intervention_required:
            self.on_intervention_required(request)
        
        return request
    
    def process_human_decision(
        self,
        freeze_id: str,
        decision: str,  # "approve", "reject", "modify"
        feedback: Optional[str] = None,
        modified_action: Optional[Dict[str, Any]] = None,
        rationale: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process human decision on pending action
        Implements Human Judgment as Normative Cognitive Event
        """
        if freeze_id not in self.frozen_states:
            return {"error": f"Frozen state not found: {freeze_id}"}
        
        frozen = self.frozen_states[freeze_id]
        
        # Determine event type
        if decision == "approve":
            event_type = HITLEventType.APPROVED
        elif decision == "reject":
            event_type = HITLEventType.REJECTED
        elif decision == "modify":
            event_type = HITLEventType.MODIFIED
        else:
            return {"error": f"Invalid decision: {decision}"}
        
        # Log human decision
        trace = self._log_trace(
            event_type=event_type,
            freeze_id=freeze_id,
            pending_action=frozen.pending_action,
            human_decision=decision,
            human_feedback=feedback,
            modified_action=modified_action,
            decision_rationale=rationale,
            actor="human"
        )
        
        print(f"\n👤 [HITL] Human decision: {decision.upper()}")
        if feedback:
            print(f"    Feedback: {feedback}")
        if rationale:
            print(f"    Rationale: {rationale}")
        
        return {
            "status": "processed",
            "decision": decision,
            "trace_id": trace.trace_id,
            "freeze_id": freeze_id,
            "next_action": self._determine_next_action(decision, frozen, modified_action)
        }
    
    def _determine_next_action(
        self,
        decision: str,
        frozen: FrozenCognitiveState,
        modified_action: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine the next action based on human decision"""
        if decision == "approve":
            return {
                "action": "execute",
                "proposed_action": frozen.pending_action
            }
        elif decision == "modify":
            return {
                "action": "execute",
                "proposed_action": modified_action or frozen.pending_action
            }
        else:  # reject
            return {
                "action": "retry_cognition",
                "rejection_feedback": frozen.cognition_output.get("reasoning", ""),
                "virtual_rejection_cycle": True
            }
    
    def create_virtual_rejection_cycle(
        self,
        frozen: FrozenCognitiveState,
        rejection_reason: str
    ) -> Dict[str, Any]:
        """
        Create a virtual rejection cycle for re-cognition
        Implements Virtual Rejection Cycles
        """
        return {
            "cycle_type": "virtual_rejection",
            "original_cognition": frozen.cognition_output,
            "rejection_reason": rejection_reason,
            "context_update": {
                "human_rejected": True,
                "rejection_reason": rejection_reason,
                "previous_proposal": frozen.pending_action,
                "retry_guidance": "Consider alternative approaches based on human feedback"
            },
            "memory_state": frozen.memory_snapshot,
            "evidence_cache": frozen.evidence_cache
        }
    
    def _log_trace(
        self,
        event_type: HITLEventType,
        freeze_id: Optional[str],
        pending_action: Dict[str, Any],
        human_decision: Optional[str] = None,
        human_feedback: Optional[str] = None,
        modified_action: Optional[Dict[str, Any]] = None,
        decision_rationale: Optional[str] = None,
        actor: str = "system"
    ) -> HITLTrace:
        """Log HITL event to Glassbox Trace"""
        trace = HITLTrace(
            trace_id=self._generate_trace_id(),
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            freeze_id=freeze_id,
            pending_action=pending_action,
            human_decision=human_decision,
            human_feedback=human_feedback,
            modified_action=modified_action,
            decision_rationale=decision_rationale,
            actor=actor
        )
        
        self.hitl_traces.append(trace)
        return trace
    
    def get_audit_log(self) -> Dict[str, Any]:
        """Get complete HITL audit log for Glassbox Trace"""
        return {
            "hitl_events": [t.to_dict() for t in self.hitl_traces],
            "frozen_states": {
                fid: fs.to_dict() for fid, fs in self.frozen_states.items()
            },
            "statistics": {
                "total_interventions": len(self.hitl_traces),
                "approvals": sum(1 for t in self.hitl_traces if t.event_type == HITLEventType.APPROVED),
                "rejections": sum(1 for t in self.hitl_traces if t.event_type == HITLEventType.REJECTED),
                "modifications": sum(1 for t in self.hitl_traces if t.event_type == HITLEventType.MODIFIED),
                "frozen_states_count": len(self.frozen_states)
            }
        }
    
    def cleanup_frozen_state(self, freeze_id: str):
        """Remove frozen state after processing (optional cleanup)"""
        if freeze_id in self.frozen_states:
            del self.frozen_states[freeze_id]


class InteractiveHITLHandler:
    """
    Interactive command-line handler for HITL decisions
    For demonstration and testing purposes
    """
    
    def __init__(self, hitl_manager: HITLManager, auto_approve: bool = False):
        self.hitl_manager = hitl_manager
        self.auto_approve = auto_approve
    
    def handle_intervention(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle intervention request interactively or automatically"""
        if self.auto_approve:
            return self._auto_approve(request)
        else:
            return self._interactive_prompt(request)
    
    def _auto_approve(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically approve for testing"""
        print(f"\n🤖 [AUTO-HITL] Auto-approving: {request['pending_action'].get('tool_name', 'N/A')}")
        return self.hitl_manager.process_human_decision(
            freeze_id=request["freeze_id"],
            decision="approve",
            rationale="Auto-approved for testing"
        )
    
    def _interactive_prompt(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Interactive command-line prompt for human decision"""
        print("\n" + "="*60)
        print("🔔 HUMAN INTERVENTION REQUIRED")
        print("="*60)
        print(f"\nReason: {request['intervention_reason']}")
        print(f"Level: {request['intervention_level']}")
        print(f"\nPending Action:")
        print(f"  Tool: {request['pending_action'].get('tool_name', 'N/A')}")
        print(f"  Parameters: {json.dumps(request['pending_action'].get('parameters', {}), indent=4)}")
        print(f"\nReasoning: {request['cognition_reasoning'][:300]}...")
        print(f"\nEvidence: {request['evidence_refs']}")
        print("\nOptions: [A]pprove, [R]eject, [M]odify")
        
        while True:
            choice = input("\nYour decision (A/R/M): ").strip().upper()
            
            if choice == 'A':
                rationale = input("Rationale (optional): ").strip() or None
                return self.hitl_manager.process_human_decision(
                    freeze_id=request["freeze_id"],
                    decision="approve",
                    rationale=rationale
                )
            
            elif choice == 'R':
                feedback = input("Rejection reason: ").strip()
                return self.hitl_manager.process_human_decision(
                    freeze_id=request["freeze_id"],
                    decision="reject",
                    feedback=feedback,
                    rationale="Human rejected action"
                )
            
            elif choice == 'M':
                print("Enter modified parameters (JSON format):")
                try:
                    mod_params = json.loads(input())
                    modified_action = {
                        "tool_name": request['pending_action'].get('tool_name'),
                        "parameters": mod_params
                    }
                    return self.hitl_manager.process_human_decision(
                        freeze_id=request["freeze_id"],
                        decision="modify",
                        modified_action=modified_action,
                        rationale="Human modified action parameters"
                    )
                except json.JSONDecodeError:
                    print("Invalid JSON. Try again.")
            
            else:
                print("Invalid choice. Enter A, R, or M.")
