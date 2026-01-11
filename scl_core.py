"""
Structured Cognitive Loop (SCL) Core Implementation with HITL Integration
Demonstrates the Retrieval-Cognition-Control-Action-Memory (R-CCAM) loop
with Context-Aware Human-in-the-Loop capabilities

Based on papers:
1. "Structured Cognitive Loop: Bridging Symbolic Control and Neural Reasoning in LLM Agents"
2. "Beyond Static Interrupts: Context-Aware Human-in-the-Loop as a Cognitive Process for Trustworthy LLM Agents"
"""

import json
import time
import copy
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from hitl_module import (
    HITLManager, HITLPolicy, InterventionLevel, 
    FrozenCognitiveState, InteractiveHITLHandler
)


class ModuleType(Enum):
    RETRIEVAL = "Retrieval"
    COGNITION = "Cognition"
    CONTROL = "Control"
    ACTION = "Action"
    MEMORY = "Memory"
    HITL = "HITL"  # New module type for HITL events


@dataclass
class LoopTrace:
    """Record of a single CCAM loop iteration"""
    loop_id: str
    timestamp: str
    module: str
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    decision: Optional[str] = None
    validation_result: Optional[bool] = None
    evidence_refs: Optional[List[str]] = None
    hitl_event: Optional[Dict[str, Any]] = None  # New field for HITL events


class MetaPrompt:
    """
    Soft Symbolic Control Layer
    Provides persistent governance constraints across all Cognition cycles
    """
    
    def __init__(self):
        self.rules = {
            "must_cite_stored_evidence": True,
            "no_final_answer_without_control_pass": True,
            "single_final_action": True,
            "avoid_redundant_tool_calls": True,
            "validate_conditional_branches": True,
            # HITL-related rules
            "require_hitl_for_high_risk": True,
            "log_all_human_decisions": True,
        }
        
        self.instructions = """
        You are operating under Soft Symbolic Control within a Structured Cognitive Loop.
        
        MANDATORY CONSTRAINTS:
        1. Always consult Memory before proposing actions
        2. Cite evidence from Retrieval/Memory in all reasoning
        3. Never execute final actions without Control validation
        4. Apply conditional logic exactly as specified
        5. Avoid redundant tool calls by checking Memory first
        6. High-risk actions require Human-in-the-Loop approval
        
        REASONING PROTOCOL:
        - State current goal explicitly
        - Reference stored evidence by ID
        - Propose action with clear rationale
        - Wait for Control validation before execution
        - Accept and incorporate human feedback when provided
        """
    
    def get_state(self) -> Dict[str, Any]:
        """Return current Metaprompt state for freezing"""
        return {
            "rules": copy.deepcopy(self.rules),
            "instructions": self.instructions
        }
    
    def validate(self, cognition_output: Dict[str, Any]) -> tuple[bool, str]:
        """Validate Cognition output against symbolic rules"""
        issues = []
        
        # Check evidence citation
        if self.rules["must_cite_stored_evidence"]:
            if not cognition_output.get("evidence_refs"):
                issues.append("Missing evidence citations")
        
        # Check final action constraint
        if cognition_output.get("is_final_action"):
            if not cognition_output.get("control_validated"):
                issues.append("Final action without Control validation")
        
        is_valid = len(issues) == 0
        message = "PASS" if is_valid else f"VIOLATIONS: {'; '.join(issues)}"
        
        return is_valid, message


class Memory:
    """
    Externalized Working Store
    Maintains state persistence across loops with audit trail
    Enhanced with snapshot/restore capabilities for HITL
    """
    
    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.history: List[LoopTrace] = []
        self.evidence_cache: Dict[str, Any] = {}
        
    def write(self, key: str, value: Any, evidence_id: Optional[str] = None):
        """Store state with optional evidence reference"""
        self.store[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "evidence_id": evidence_id
        }
        
    def read(self, key: str) -> Optional[Any]:
        """Retrieve stored state"""
        entry = self.store.get(key)
        return entry["value"] if entry else None
    
    def has_evidence(self, evidence_id: str) -> bool:
        """Check if evidence already exists (avoid redundant calls)"""
        return evidence_id in self.evidence_cache
    
    def store_evidence(self, evidence_id: str, data: Any):
        """Cache retrieved evidence"""
        self.evidence_cache[evidence_id] = data
    
    def get_evidence(self, evidence_id: str) -> Optional[Any]:
        """Retrieve cached evidence"""
        return self.evidence_cache.get(evidence_id)
    
    def log_trace(self, trace: LoopTrace):
        """Append to audit log"""
        self.history.append(trace)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Return current state for Cognition context"""
        return {
            "stored_values": {k: v["value"] for k, v in self.store.items()},
            "available_evidence": list(self.evidence_cache.keys()),
            "loop_count": len(self.history)
        }
    
    # HITL-related methods for state freezing/thawing
    def get_snapshot(self) -> Dict[str, Any]:
        """Get complete memory snapshot for state freezing"""
        return {
            "store": copy.deepcopy(self.store),
            "history_length": len(self.history)
        }
    
    def restore_snapshot(self, snapshot: Dict[str, Any]):
        """Restore memory from snapshot (for state thawing)"""
        self.store = copy.deepcopy(snapshot.get("store", {}))
    
    def get_evidence_snapshot(self) -> Dict[str, Any]:
        """Get evidence cache snapshot"""
        return copy.deepcopy(self.evidence_cache)
    
    def restore_evidence(self, evidence_snapshot: Dict[str, Any]):
        """Restore evidence cache from snapshot"""
        self.evidence_cache = copy.deepcopy(evidence_snapshot)


class ToolRegistry:
    """Registry of available tools for Action module"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        
    def register(self, name: str, func: Callable, description: str):
        """Register a tool with metadata"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "name": name
        }
    
    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Return list of available tools for Cognition"""
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self.tools.values()
        ]
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not registered")
        
        func = self.tools[tool_name]["function"]
        return func(**kwargs)


class StructuredCognitiveLoopWithHITL:
    """
    Structured Cognitive Loop (SCL) with Human-in-the-Loop Integration
    
    Implements:
    - R-CCAM loop with Soft Symbolic Control
    - Context-Aware Human-in-the-Loop
    - Cognitive State Freezing and Thawing
    - Virtual Rejection Cycles
    - Trace-Grounded Resumption
    - Glassbox Trace for full auditability
    """
    
    def __init__(
        self,
        cognition_engine: Callable,
        tool_registry: ToolRegistry,
        metaprompt: Optional[MetaPrompt] = None,
        hitl_policy: Optional[HITLPolicy] = None,
        max_loops: int = 20,
        hitl_mode: str = "interactive"  # "interactive", "auto", "disabled"
    ):
        self.cognition_engine = cognition_engine
        self.tools = tool_registry
        self.metaprompt = metaprompt or MetaPrompt()
        self.memory = Memory()
        self.max_loops = max_loops
        self.loop_counter = 0
        
        # HITL components
        self.hitl_manager = HITLManager(policy=hitl_policy)
        self.hitl_mode = hitl_mode
        
        # Create appropriate HITL handler based on mode
        if hitl_mode == "auto":
            self.hitl_handler = InteractiveHITLHandler(self.hitl_manager, auto_approve=True)
        elif hitl_mode == "interactive":
            self.hitl_handler = InteractiveHITLHandler(self.hitl_manager, auto_approve=False)
        else:
            self.hitl_handler = None
        
        # Track if we're in a resumed state
        self._is_resumed = False
        self._resume_context = None
        
    def retrieval(self, task: str) -> Dict[str, Any]:
        """
        Retrieval Module (invoked once at task start)
        Performs initial evidence gathering and task decomposition
        """
        print(f"\n{'='*60}")
        print(f"[RETRIEVAL] Initializing task: {task[:100]}...")
        print(f"{'='*60}\n")
        
        # Simulate retrieval planning
        plan = {
            "evidence_needed": ["SF_weather", "Miami_weather", "Atlanta_weather"],
            "base_temperature": 55,
            "conditions_parsed": True,
            "tools_required": ["get_weather", "send_email", "generate_image", "cancel_trip"]
        }
        
        self.memory.write("task", task)
        self.memory.write("retrieval_plan", plan)
        
        trace = LoopTrace(
            loop_id="R-001",
            timestamp=datetime.now().isoformat(),
            module=ModuleType.RETRIEVAL.value,
            input_state={"task": task},
            output_state=plan
        )
        self.memory.log_trace(trace)
        
        return plan
    
    def cognition(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cognition Module (probabilistic inference under symbolic constraints)
        Enhanced with HITL feedback integration
        """
        self.loop_counter += 1
        loop_id = f"CCAM-{self.loop_counter:03d}"
        
        print(f"\n[COGNITION] Loop {self.loop_counter}")
        print(f"{'─'*60}")
        
        # Check for human rejection feedback (Virtual Rejection Cycle)
        if context.get("human_rejected"):
            print(f"📝 Incorporating human feedback: {context.get('rejection_reason', 'N/A')}")
        
        state_summary = self.memory.get_state_summary()
        available_tools = self.tools.get_tool_descriptions()
        
        # Enhanced prompt with HITL context
        hitl_context = ""
        if context.get("human_rejected"):
            hitl_context = f"""
            HUMAN FEEDBACK (incorporate this):
            - Previous proposal was rejected
            - Reason: {context.get('rejection_reason', 'Not specified')}
            - Guidance: {context.get('retry_guidance', 'Consider alternative approaches')}
            """
        
        cognition_prompt = f"""
        {self.metaprompt.instructions}
        
        {hitl_context}
        
        CURRENT STATE:
        {json.dumps(state_summary, indent=2)}
        
        AVAILABLE TOOLS:
        {json.dumps(available_tools, indent=2)}
        
        CONTEXT:
        {json.dumps(context, indent=2)}
        
        Based on the above, determine:
        1. What is the next action needed?
        2. What evidence supports this decision?
        3. Are all conditions for this action met?
        
        Respond in JSON format with:
        - reasoning: your thought process
        - proposed_action: {{tool_name, parameters}}
        - evidence_refs: list of evidence IDs used
        - is_final_action: boolean
        - confidence: float between 0 and 1
        """
        
        response = self.cognition_engine(cognition_prompt, context)
        
        print(f"Reasoning: {response.get('reasoning', 'N/A')}")
        print(f"Proposed Action: {response.get('proposed_action', 'N/A')}")
        
        trace = LoopTrace(
            loop_id=loop_id,
            timestamp=datetime.now().isoformat(),
            module=ModuleType.COGNITION.value,
            input_state=context,
            output_state=response,
            evidence_refs=response.get("evidence_refs")
        )
        self.memory.log_trace(trace)
        
        return response
    
    def control(self, cognition_output: Dict[str, Any]) -> tuple[bool, str]:
        """
        Control Module (Soft Symbolic Validation)
        Validates Cognition output against Metaprompt rules
        """
        print(f"\n[CONTROL] Validating proposed action...")
        
        if cognition_output.get("is_final_action"):
            cognition_output["control_validated"] = True
        
        is_valid, message = self.metaprompt.validate(cognition_output)
        
        proposed_action = cognition_output.get("proposed_action", {})
        tool_name = proposed_action.get("tool_name")
        
        if tool_name:
            evidence_id = f"evidence_{tool_name}_{json.dumps(proposed_action.get('parameters', {}), sort_keys=True)}"
            if self.memory.has_evidence(evidence_id):
                is_valid = False
                message = "REJECTED: Redundant tool call (evidence already in Memory)"
        
        trace = LoopTrace(
            loop_id=f"CTL-{self.loop_counter:03d}",
            timestamp=datetime.now().isoformat(),
            module=ModuleType.CONTROL.value,
            input_state=cognition_output,
            output_state={"validation": is_valid, "message": message},
            validation_result=is_valid
        )
        self.memory.log_trace(trace)
        
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{status}: {message}")
        
        return is_valid, message
    
    def hitl_check(
        self, 
        cognition_output: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        HITL Check Module (Action-Centric Intervention)
        Determines if human intervention is required and handles the interaction
        
        Returns:
            (should_proceed, modified_action_or_none)
        """
        if self.hitl_mode == "disabled":
            return True, None
        
        # Check if intervention is required
        intervention_level, reason = self.hitl_manager.check_intervention(
            cognition_output, context, self.loop_counter
        )
        
        if intervention_level == InterventionLevel.NONE:
            return True, None
        
        print(f"\n[HITL] Intervention required: {intervention_level.value}")
        print(f"       Reason: {reason}")
        
        # Freeze cognitive state
        frozen_state = self.hitl_manager.freeze_state(
            loop_counter=self.loop_counter,
            cognition_output=cognition_output,
            memory_state=self.memory.get_snapshot(),
            evidence_cache=self.memory.get_evidence_snapshot(),
            context=context,
            metaprompt_state=self.metaprompt.get_state(),
            intervention_level=intervention_level,
            intervention_reason=reason
        )
        
        # Request approval
        request = self.hitl_manager.request_approval(frozen_state)
        
        # Handle the intervention
        if self.hitl_handler:
            result = self.hitl_handler.handle_intervention(request)
        else:
            # If no handler, auto-approve for NOTIFY level, otherwise block
            if intervention_level == InterventionLevel.NOTIFY:
                result = self.hitl_manager.process_human_decision(
                    frozen_state.freeze_id, "approve", rationale="Auto-approved (notify level)"
                )
            else:
                print("⚠️  No HITL handler available. Blocking execution.")
                return False, None
        
        # Log HITL event
        hitl_trace = LoopTrace(
            loop_id=f"HITL-{self.loop_counter:03d}",
            timestamp=datetime.now().isoformat(),
            module=ModuleType.HITL.value,
            input_state={"freeze_id": frozen_state.freeze_id, "level": intervention_level.value},
            output_state=result,
            hitl_event=result
        )
        self.memory.log_trace(hitl_trace)
        
        # Process result
        next_action = result.get("next_action", {})
        
        if next_action.get("action") == "execute":
            # Approved or modified - thaw state and proceed
            self.hitl_manager.thaw_state(frozen_state.freeze_id)
            modified = next_action.get("proposed_action")
            if modified != cognition_output.get("proposed_action"):
                return True, modified
            return True, None
        
        elif next_action.get("action") == "retry_cognition":
            # Rejected - need to re-enter cognition with feedback
            self.hitl_manager.thaw_state(frozen_state.freeze_id)
            
            # Create virtual rejection cycle
            rejection_cycle = self.hitl_manager.create_virtual_rejection_cycle(
                frozen_state,
                result.get("feedback", "Human rejected the action")
            )
            
            # Store rejection context for next cognition cycle
            self._resume_context = rejection_cycle.get("context_update", {})
            return False, {"virtual_rejection": True}
        
        return False, None
    
    def action(self, cognition_output: Dict[str, Any], modified_action: Optional[Dict[str, Any]] = None) -> Any:
        """
        Action Module (Separated Execution)
        Executes validated actions, with support for HITL-modified actions
        """
        print(f"\n[ACTION] Executing validated action...")
        
        # Use modified action if provided by HITL
        proposed_action = modified_action or cognition_output.get("proposed_action", {})
        tool_name = proposed_action.get("tool_name")
        parameters = proposed_action.get("parameters", {})
        
        if not tool_name:
            return {"status": "no_action", "result": None}
        
        try:
            result = self.tools.execute(tool_name, **parameters)
            
            evidence_id = f"evidence_{tool_name}_{json.dumps(parameters, sort_keys=True)}"
            self.memory.store_evidence(evidence_id, result)
            
            print(f"Executed: {tool_name}")
            print(f"Result: {str(result)[:200]}...")
            
            trace = LoopTrace(
                loop_id=f"ACT-{self.loop_counter:03d}",
                timestamp=datetime.now().isoformat(),
                module=ModuleType.ACTION.value,
                input_state=proposed_action,
                output_state={"result": result, "evidence_id": evidence_id}
            )
            self.memory.log_trace(trace)
            
            return result
            
        except Exception as e:
            error_msg = f"Action execution failed: {str(e)}"
            print(f"✗ ERROR: {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Main execution loop: Retrieval → CCAM cycles with HITL → Final result
        
        Enhanced flow:
        R → [C → Control → HITL Check → A → M] → ... → Complete
        """
        print(f"\n{'#'*60}")
        print(f"# STRUCTURED COGNITIVE LOOP (SCL) WITH HITL")
        print(f"# Mode: {self.hitl_mode}")
        print(f"{'#'*60}")
        
        # Step 1: Retrieval (once)
        retrieval_result = self.retrieval(task)
        
        # Step 2: CCAM Loop with HITL
        context = {
            "task": task,
            "retrieval_plan": retrieval_result,
            "status": "in_progress"
        }
        
        while self.loop_counter < self.max_loops:
            # Check for resume context from virtual rejection cycle
            if self._resume_context:
                context.update(self._resume_context)
                self._resume_context = None
            
            # Cognition
            cognition_output = self.cognition(context)
            
            # Clear human rejection flag after cognition uses it
            context.pop("human_rejected", None)
            context.pop("rejection_reason", None)
            context.pop("retry_guidance", None)
            
            # Control
            is_valid, validation_msg = self.control(cognition_output)
            
            if not is_valid:
                print(f"\n⚠️  Control rejected action. Re-entering Cognition...")
                context["last_rejection"] = validation_msg
                continue
            
            # HITL Check (Action-Centric Intervention)
            should_proceed, modified_action = self.hitl_check(cognition_output, context)
            
            if not should_proceed:
                if modified_action and modified_action.get("virtual_rejection"):
                    # Virtual rejection cycle - re-enter cognition
                    print(f"\n🔄 [HITL] Virtual rejection cycle - re-entering Cognition...")
                    continue
                else:
                    print(f"\n⛔ [HITL] Execution blocked. Awaiting human decision...")
                    break
            
            # Action (with possible modification from HITL)
            action_result = self.action(cognition_output, modified_action)
            
            # Update context
            context["last_action_result"] = action_result
            
            # Check completion
            if cognition_output.get("is_final_action"):
                print(f"\n{'='*60}")
                print(f"[COMPLETION] Task finished in {self.loop_counter} loops")
                print(f"{'='*60}\n")
                break
        
        return self._generate_audit_report()
    
    def resume_from_freeze(self, freeze_id: str, decision: str, **kwargs) -> Dict[str, Any]:
        """
        Resume execution from a frozen state (Trace-Grounded Resumption)
        
        Args:
            freeze_id: ID of the frozen state
            decision: "approve", "reject", or "modify"
            **kwargs: Additional arguments (feedback, modified_action, rationale)
        
        Returns:
            Audit report after resumed execution
        """
        print(f"\n{'#'*60}")
        print(f"# RESUMING FROM FROZEN STATE: {freeze_id}")
        print(f"{'#'*60}")
        
        # Process the human decision
        result = self.hitl_manager.process_human_decision(
            freeze_id=freeze_id,
            decision=decision,
            **kwargs
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        # Get the frozen state
        frozen = self.hitl_manager.frozen_states.get(freeze_id)
        if not frozen:
            return {"error": "Frozen state not found after processing"}
        
        # Restore state
        self.memory.restore_snapshot(frozen.memory_snapshot)
        self.memory.restore_evidence(frozen.evidence_cache)
        self.loop_counter = frozen.loop_counter
        
        # Thaw the state
        self.hitl_manager.thaw_state(freeze_id)
        
        # Determine next action
        next_action = result.get("next_action", {})
        context = frozen.context
        
        if next_action.get("action") == "execute":
            # Execute the (possibly modified) action
            modified = next_action.get("proposed_action")
            action_result = self.action(frozen.cognition_output, modified)
            context["last_action_result"] = action_result
            
            # Continue the loop if not final
            if not frozen.cognition_output.get("is_final_action"):
                self._is_resumed = True
                return self._continue_execution(context)
            else:
                return self._generate_audit_report()
        
        elif next_action.get("action") == "retry_cognition":
            # Create virtual rejection cycle and continue
            context.update({
                "human_rejected": True,
                "rejection_reason": kwargs.get("feedback", "Action rejected by human"),
                "retry_guidance": "Consider alternative approaches based on human feedback"
            })
            self._is_resumed = True
            return self._continue_execution(context)
        
        return self._generate_audit_report()
    
    def _continue_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Continue execution from current state"""
        while self.loop_counter < self.max_loops:
            cognition_output = self.cognition(context)
            
            context.pop("human_rejected", None)
            context.pop("rejection_reason", None)
            context.pop("retry_guidance", None)
            
            is_valid, validation_msg = self.control(cognition_output)
            
            if not is_valid:
                context["last_rejection"] = validation_msg
                continue
            
            should_proceed, modified_action = self.hitl_check(cognition_output, context)
            
            if not should_proceed:
                if modified_action and modified_action.get("virtual_rejection"):
                    continue
                else:
                    break
            
            action_result = self.action(cognition_output, modified_action)
            context["last_action_result"] = action_result
            
            if cognition_output.get("is_final_action"):
                print(f"\n{'='*60}")
                print(f"[COMPLETION] Task finished in {self.loop_counter} loops")
                print(f"{'='*60}\n")
                break
        
        return self._generate_audit_report()
    
    def _generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit log including HITL events (Glassbox Trace)"""
        hitl_audit = self.hitl_manager.get_audit_log()
        
        report = {
            "task": self.memory.read("task"),
            "policies": list(self.metaprompt.rules.keys()),
            "hitl_mode": self.hitl_mode,
            "log": [asdict(trace) for trace in self.memory.history],
            "hitl_log": hitl_audit,
            "summary": {
                "total_loops": self.loop_counter,
                "policy_violations": sum(
                    1 for t in self.memory.history 
                    if t.validation_result is False
                ),
                "hitl_interventions": hitl_audit["statistics"]["total_interventions"],
                "hitl_approvals": hitl_audit["statistics"]["approvals"],
                "hitl_rejections": hitl_audit["statistics"]["rejections"],
                "final_state": self.memory.get_state_summary()
            }
        }
        return report


# Backward compatibility alias
StructuredCognitiveLoop = StructuredCognitiveLoopWithHITL


def save_audit_log(report: Dict[str, Any], filename: str = "execution_audit.json"):
    """Save audit log to JSON file"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Audit log saved to {filename}")
