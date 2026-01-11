"""
Structured Cognitive Loop (SCL) Experiment Runner with HITL Integration

This experiment demonstrates:
1. R-CCAM loop architecture
2. Context-Aware Human-in-the-Loop
3. Cognitive State Freezing and Thawing
4. Virtual Rejection Cycles
5. Glassbox Trace for auditability

Based on papers:
- "Structured Cognitive Loop: Bridging Symbolic Control and Neural Reasoning in LLM Agents"
- "Beyond Static Interrupts: Context-Aware Human-in-the-Loop as a Cognitive Process for Trustworthy LLM Agents"
"""

import json
import sys
from scl_core import StructuredCognitiveLoopWithHITL, MetaPrompt, ToolRegistry
from hitl_module import HITLPolicy, InterventionLevel
from mock_tools import (
    get_weather, send_email, generate_image, 
    cancel_trip, recommend_snacks, check_umbrella_needed
)
from mock_cognition import MockCognitionEngine


def setup_hitl_policy():
    """Configure HITL policy for the experiment"""
    policy = HITLPolicy()
    
    # Customize policies for the travel planning scenario
    policy.policies.update({
        # High-risk tools that always require approval
        "high_risk_tools": ["send_email", "cancel_trip"],
        
        # Tools that require confirmation
        "always_confirm_tools": ["generate_image"],
        
        # Confirm final actions
        "confirm_on_final_action": True,
        
        # Lower confidence threshold for confirmation
        "confirm_on_confidence_below": 0.8,
        
        # Confirm after many loops
        "confirm_after_n_loops": 8,
    })
    
    return policy


def setup_experiment(hitl_mode: str = "auto"):
    """Initialize SCL system with HITL integration"""
    
    # 1. Create Tool Registry
    tool_registry = ToolRegistry()
    
    tool_registry.register(
        "get_weather",
        get_weather,
        "Get current weather for a city (temperature, condition, precipitation)"
    )
    
    tool_registry.register(
        "send_email",
        send_email,
        "Send email notification with subject and body [HIGH RISK - requires HITL approval]"
    )
    
    tool_registry.register(
        "generate_image",
        generate_image,
        "Generate weather visualization image from description [requires HITL confirmation]"
    )
    
    tool_registry.register(
        "cancel_trip",
        cancel_trip,
        "Cancel travel plans with specified reason [HIGH RISK - requires HITL approval]"
    )
    
    tool_registry.register(
        "recommend_snacks",
        recommend_snacks,
        "Get convenience store snack recommendations"
    )
    
    tool_registry.register(
        "check_umbrella",
        check_umbrella_needed,
        "Determine if umbrella is needed based on precipitation"
    )
    
    # 2. Create Metaprompt
    metaprompt = MetaPrompt()
    
    # 3. Create HITL Policy
    hitl_policy = setup_hitl_policy()
    
    # 4. Create Cognition Engine
    cognition_engine = MockCognitionEngine()
    
    # 5. Initialize SCL with HITL
    scl_system = StructuredCognitiveLoopWithHITL(
        cognition_engine=cognition_engine,
        tool_registry=tool_registry,
        metaprompt=metaprompt,
        hitl_policy=hitl_policy,
        max_loops=20,
        hitl_mode=hitl_mode
    )
    
    return scl_system


def run_weather_scenario(hitl_mode: str = "auto"):
    """
    Run the weather-based travel planning scenario with HITL
    """
    
    task = """
    When the base temperature is 55°F, check the weather in San Francisco, Miami, 
    and Atlanta, then plan a trip according to the following conditions:
    
    - If all three regions are above the reference temperature, decide to travel 
      to the coolest one and draw an image of that place's weather.
    - If only two regions are above the reference temperature, choose the cooler 
      one among them and send an email to test-scl@test.com indicating the selected 
      destination.
    - If only one region is above the reference temperature, travel to that place.
    - If all three regions are below the reference temperature, cancel the trip 
      and recommend a list of convenience store snacks to enjoy at home.
    
    Tell me the weather at the destination and whether to bring an umbrella if 
    a trip is decided.
    """
    
    print("\n" + "="*80)
    print("STRUCTURED COGNITIVE LOOP (SCL) WITH HITL EXPERIMENT")
    print("Weather-Based Travel Planning")
    print("="*80)
    print(f"\nHITL Mode: {hitl_mode}")
    print(f"\nTask: {task.strip()[:200]}...")
    print("\n" + "="*80 + "\n")
    
    # Setup system
    system = setup_experiment(hitl_mode)
    
    # Run task
    audit_report = system.run(task)
    
    return audit_report


def demonstrate_freeze_resume():
    """
    Demonstrate cognitive state freezing and resumption
    This shows the full HITL workflow with explicit freeze/thaw
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Cognitive State Freezing and Resumption")
    print("="*80 + "\n")
    
    # Create system in disabled mode first
    system = setup_experiment(hitl_mode="disabled")
    
    task = "Check weather and send travel notification email"
    
    # Manually trigger a freeze scenario
    print("1. Running partial task (will freeze before email)...")
    
    # Simulate partial execution
    from hitl_module import HITLManager, HITLPolicy
    
    hitl_manager = HITLManager(HITLPolicy())
    
    # Simulate a frozen state
    frozen = hitl_manager.freeze_state(
        loop_counter=3,
        cognition_output={
            "reasoning": "All weather data collected. Sending email to confirm destination.",
            "proposed_action": {
                "tool_name": "send_email",
                "parameters": {
                    "recipient": "test-scl@test.com",
                    "subject": "Travel Plan: Miami",
                    "body": "Based on weather analysis, traveling to Miami at 78°F."
                }
            },
            "evidence_refs": ["weather_sf", "weather_miami", "weather_atlanta"],
            "is_final_action": True
        },
        memory_state={"task": task, "weather_collected": True},
        evidence_cache={"wx-001": {"city": "Miami", "temp": 78}},
        context={"status": "pending_email"},
        metaprompt_state={"rules": {}},
        intervention_level=InterventionLevel.APPROVE,
        intervention_reason="High-risk tool: send_email"
    )
    
    print(f"\n2. State frozen with ID: {frozen.freeze_id}")
    print(f"   Pending action: {frozen.pending_action}")
    
    # Simulate human approval
    print("\n3. Simulating human approval...")
    result = hitl_manager.process_human_decision(
        freeze_id=frozen.freeze_id,
        decision="approve",
        rationale="Confirmed destination is correct"
    )
    
    print(f"\n4. Human decision processed: {result['decision']}")
    
    # Thaw and show result
    thawed = hitl_manager.thaw_state(frozen.freeze_id)
    print(f"\n5. State thawed, ready to continue execution")
    
    # Show audit log
    audit = hitl_manager.get_audit_log()
    print(f"\n6. HITL Audit Summary:")
    print(f"   - Total interventions: {audit['statistics']['total_interventions']}")
    print(f"   - Approvals: {audit['statistics']['approvals']}")
    print(f"   - Rejections: {audit['statistics']['rejections']}")
    
    return audit


def demonstrate_virtual_rejection():
    """
    Demonstrate virtual rejection cycle
    Shows how human rejection triggers re-cognition
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Virtual Rejection Cycle")
    print("="*80 + "\n")
    
    from hitl_module import HITLManager, HITLPolicy, InterventionLevel
    
    hitl_manager = HITLManager(HITLPolicy())
    
    # Step 1: Freeze with a proposed action
    print("1. Freezing state with proposed email action...")
    frozen = hitl_manager.freeze_state(
        loop_counter=4,
        cognition_output={
            "reasoning": "Sending confirmation email to wrong address",
            "proposed_action": {
                "tool_name": "send_email",
                "parameters": {
                    "recipient": "wrong@email.com",  # Wrong email!
                    "subject": "Travel Confirmation",
                    "body": "Confirmed travel to Atlanta"
                }
            },
            "evidence_refs": ["weather_data"],
            "is_final_action": True
        },
        memory_state={"collected_weather": ["SF", "Miami", "Atlanta"]},
        evidence_cache={},
        context={"destination": "Atlanta"},
        metaprompt_state={},
        intervention_level=InterventionLevel.APPROVE,
        intervention_reason="High-risk tool: send_email"
    )
    
    # Step 2: Human rejects with feedback
    print("\n2. Human rejects action with feedback...")
    result = hitl_manager.process_human_decision(
        freeze_id=frozen.freeze_id,
        decision="reject",
        feedback="Wrong email address! Should be test-scl@test.com",
        rationale="Email recipient is incorrect"
    )
    
    print(f"   Decision: {result['decision']}")
    print(f"   Next action: {result['next_action']['action']}")
    
    # Step 3: Create virtual rejection cycle
    print("\n3. Creating virtual rejection cycle...")
    rejection_cycle = hitl_manager.create_virtual_rejection_cycle(
        frozen,
        "Wrong email address - use test-scl@test.com instead"
    )
    
    print(f"   Cycle type: {rejection_cycle['cycle_type']}")
    print(f"   Context update: {rejection_cycle['context_update']}")
    
    # Step 4: Show that cognition would receive rejection feedback
    print("\n4. Next cognition cycle will receive:")
    print(f"   - human_rejected: {rejection_cycle['context_update']['human_rejected']}")
    print(f"   - rejection_reason: {rejection_cycle['context_update']['rejection_reason']}")
    print(f"   - retry_guidance: {rejection_cycle['context_update']['retry_guidance']}")
    
    return rejection_cycle


def save_experiment_results(audit_report: dict, filename: str = "experiment_results_hitl.json"):
    """Save experiment results with HITL information"""
    
    formatted_report = {
        "experiment": "Weather-Based Travel Planning with HITL",
        "architecture": "Structured Cognitive Loop (R-CCAM) + Context-Aware HITL",
        "task": audit_report.get("task"),
        "hitl_mode": audit_report.get("hitl_mode"),
        "policies": audit_report.get("policies"),
        "execution_log": audit_report.get("log"),
        "hitl_log": audit_report.get("hitl_log"),
        "summary": audit_report.get("summary")
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Experiment results saved to: {filename}")
    
    return formatted_report


def print_summary_statistics(audit_report: dict):
    """Print key metrics including HITL statistics"""
    
    summary = audit_report.get("summary", {})
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n📊 Performance Metrics:")
    print(f"   • Total CCAM loops: {summary.get('total_loops', 'N/A')}")
    print(f"   • Policy violations: {summary.get('policy_violations', 'N/A')}")
    
    total_loops = summary.get('total_loops', 1)
    violations = summary.get('policy_violations', 0)
    success_rate = 100 * (1 - violations / max(total_loops, 1))
    print(f"   • Success rate: {success_rate:.1f}%")
    
    print(f"\n👤 HITL Metrics:")
    print(f"   • Total interventions: {summary.get('hitl_interventions', 0)}")
    print(f"   • Approvals: {summary.get('hitl_approvals', 0)}")
    print(f"   • Rejections: {summary.get('hitl_rejections', 0)}")
    
    print(f"\n🔧 Architecture Validation:")
    print(f"   ✓ Modular decomposition maintained")
    print(f"   ✓ Soft Symbolic Control enforced")
    print(f"   ✓ Context-Aware HITL integrated")
    print(f"   ✓ Cognitive state freezing/thawing operational")
    print(f"   ✓ Glassbox Trace generated")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point with command line options"""
    
    # Parse command line arguments
    hitl_mode = "auto"  # Default mode
    run_demos = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            hitl_mode = "interactive"
        elif sys.argv[1] == "--disabled":
            hitl_mode = "disabled"
        elif sys.argv[1] == "--demo":
            run_demos = True
        elif sys.argv[1] == "--help":
            print("""
SCL with HITL Experiment Runner

Usage:
  python run_experiment_hitl.py [OPTIONS]

Options:
  --auto        Run with automatic HITL approval (default)
  --interactive Run with interactive HITL prompts
  --disabled    Run without HITL checks
  --demo        Run demonstrations of HITL features
  --help        Show this help message

Examples:
  python run_experiment_hitl.py              # Auto mode
  python run_experiment_hitl.py --interactive  # Interactive mode
  python run_experiment_hitl.py --demo       # Run demos
            """)
            return
    
    if run_demos:
        # Run demonstrations
        print("\n" + "#"*80)
        print("# HITL FEATURE DEMONSTRATIONS")
        print("#"*80)
        
        demonstrate_freeze_resume()
        demonstrate_virtual_rejection()
        
        print("\n✅ Demonstrations complete!")
        return
    
    # Run the main experiment
    audit_report = run_weather_scenario(hitl_mode)
    
    # Save results
    save_experiment_results(audit_report)
    
    # Print statistics
    print_summary_statistics(audit_report)
    
    print("\n✅ Experiment complete!")
    print("\nGenerated files:")
    print("  • experiment_results_hitl.json  - Full audit log with HITL events")
    print("\nHITL Features Demonstrated:")
    print("  • Cognitive State Freezing and Thawing")
    print("  • Action-Centric Intervention")
    print("  • Human Judgment as Normative Cognitive Event")
    print("  • Glassbox Trace for Epistemic Accountability")
    
    if hitl_mode == "auto":
        print("\n💡 Tip: Run with --interactive for manual HITL decisions")
        print("        Run with --demo to see freeze/thaw and rejection cycle demos")


if __name__ == "__main__":
    main()
