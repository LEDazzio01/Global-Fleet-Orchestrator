"""
Scheduler - Makes workload placement decisions based on risk assessments.

The Scheduler is the top-level decision maker that:
1. Uses the RiskEngine to assess thermal risk
2. Determines whether to approve or block workloads
3. Provides clear decisions with reasoning

This separates the decision logic from the UI, following SOLID principles.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd

from src.core.risk_engine import RiskAssessment, RiskEngine, RiskLevel
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class Decision(str, Enum):
    """Scheduler decisions for workload placement."""
    
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"
    CONDITIONAL = "CONDITIONAL"


@dataclass
class SchedulerDecision:
    """
    Result of a scheduling decision.
    
    Contains the decision, reasoning, and supporting data.
    This is what the UI layer consumes to display results.
    """
    
    decision: Decision
    reason: str
    risk_assessment: RiskAssessment
    workload_shift_pct: float
    baseline_risk: Optional[RiskAssessment] = None
    
    @property
    def is_approved(self) -> bool:
        """Check if the decision is to approve."""
        return self.decision == Decision.APPROVED
    
    @property
    def is_blocked(self) -> bool:
        """Check if the decision is to block."""
        return self.decision == Decision.BLOCKED
    
    @property
    def risk_reduction(self) -> Optional[float]:
        """Calculate risk reduction from baseline if available."""
        if self.baseline_risk is None:
            return None
        return self.baseline_risk.max_upper_bound - self.risk_assessment.max_upper_bound
    
    @property
    def resolved_breach(self) -> bool:
        """Check if workload shift resolved a thermal breach."""
        if self.baseline_risk is None:
            return False
        return (
            self.baseline_risk.exceeds_threshold and 
            not self.risk_assessment.exceeds_threshold
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = {
            "decision": self.decision.value,
            "reason": self.reason,
            "workload_shift_pct": self.workload_shift_pct,
            "risk_assessment": self.risk_assessment.to_dict(),
        }
        if self.baseline_risk:
            result["baseline_risk"] = self.baseline_risk.to_dict()
            result["risk_reduction"] = self.risk_reduction
            result["resolved_breach"] = self.resolved_breach
        return result


class Scheduler:
    """
    Workload scheduler that makes thermal-aware placement decisions.
    
    Uses dependency injection for the RiskEngine, allowing easy testing.
    
    Example:
        engine = RiskEngine(model, region_map)
        scheduler = Scheduler(engine)
        decision = scheduler.evaluate_workload(data, "Arizona", shift_pct=20)
        if decision.is_blocked:
            print(f"Workload blocked: {decision.reason}")
    """
    
    def __init__(
        self,
        risk_engine: RiskEngine,
    ) -> None:
        """
        Initialize the Scheduler.
        
        Args:
            risk_engine: RiskEngine instance for risk assessment.
        """
        self._risk_engine = risk_engine
        logger.info("Scheduler initialized")
    
    def evaluate_workload(
        self,
        data: pd.DataFrame,
        region: str,
        workload_shift_pct: float = 0.0,
    ) -> SchedulerDecision:
        """
        Evaluate whether workload can be placed in a region.
        
        Args:
            data: DataFrame with telemetry data.
            region: Region to evaluate.
            workload_shift_pct: Percentage of workload to shift away (0-100).
        
        Returns:
            SchedulerDecision with approval/block decision and reasoning.
        """
        # Calculate temperature adjustment from workload shift
        temp_adjustment = workload_shift_pct * settings.COOLING_FACTOR
        
        # Get baseline risk (no adjustment)
        baseline_risk = self._risk_engine.assess_risk(data, region, 0.0)
        
        # Get optimized risk if shift is applied
        if workload_shift_pct > 0:
            optimized_risk = self._risk_engine.assess_risk(
                data, region, temp_adjustment
            )
        else:
            optimized_risk = baseline_risk
        
        # Make decision
        decision, reason = self._make_decision(
            baseline_risk, optimized_risk, workload_shift_pct
        )
        
        result = SchedulerDecision(
            decision=decision,
            reason=reason,
            risk_assessment=optimized_risk,
            workload_shift_pct=workload_shift_pct,
            baseline_risk=baseline_risk if workload_shift_pct > 0 else None,
        )
        
        logger.info(
            "Scheduling decision made",
            extra=result.to_dict(),
        )
        
        return result
    
    def _make_decision(
        self,
        baseline: RiskAssessment,
        optimized: RiskAssessment,
        shift_pct: float,
    ) -> tuple[Decision, str]:
        """
        Make the scheduling decision based on risk assessments.
        
        Returns:
            Tuple of (Decision, reason string).
        """
        threshold = self._risk_engine.thermal_limit
        
        # If optimized assessment is safe
        if not optimized.exceeds_threshold:
            if shift_pct > 0 and baseline.exceeds_threshold:
                # Workload shift resolved the issue
                return (
                    Decision.APPROVED,
                    f"Workload shift of {shift_pct:.0f}% resolved thermal breach. "
                    f"Risk reduced from {baseline.max_upper_bound:.1f}°C to "
                    f"{optimized.max_upper_bound:.1f}°C."
                )
            else:
                # System is within safe envelope
                return (
                    Decision.APPROVED,
                    f"Within safety envelope. Max risk: {optimized.max_upper_bound:.1f}°C "
                    f"(threshold: {threshold:.1f}°C)."
                )
        
        # System is still unsafe
        if shift_pct > 0 and optimized.max_upper_bound < baseline.max_upper_bound:
            # Partial improvement
            needed_shift = self._calculate_required_shift(optimized.max_upper_bound)
            return (
                Decision.BLOCKED,
                f"Risk reduced but still exceeds threshold. "
                f"Current: {optimized.max_upper_bound:.1f}°C > {threshold:.1f}°C. "
                f"Need approximately {needed_shift:.0f}% total shift."
            )
        else:
            # No shift or shift ineffective
            needed_shift = self._calculate_required_shift(baseline.max_upper_bound)
            return (
                Decision.BLOCKED,
                f"Upper bound {optimized.max_upper_bound:.1f}°C exceeds "
                f"safety limit {threshold:.1f}°C. "
                f"Recommend shifting {needed_shift:.0f}% of workload."
            )
    
    def _calculate_required_shift(self, current_upper_bound: float) -> float:
        """Calculate the workload shift percentage needed to reach safety."""
        threshold = self._risk_engine.thermal_limit
        if current_upper_bound <= threshold:
            return 0.0
        
        excess = current_upper_bound - threshold
        required_shift = excess / settings.COOLING_FACTOR
        return min(required_shift, 100.0)  # Cap at 100%
    
    def get_optimization_scenarios(
        self,
        data: pd.DataFrame,
        region: str,
        shift_increments: List[int] = None,
    ) -> List[SchedulerDecision]:
        """
        Evaluate multiple workload shift scenarios.
        
        Useful for UI to show the user different options.
        
        Args:
            data: DataFrame with telemetry data.
            region: Region to evaluate.
            shift_increments: List of shift percentages to evaluate.
        
        Returns:
            List of SchedulerDecisions for each scenario.
        """
        if shift_increments is None:
            shift_increments = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        scenarios = []
        for shift in shift_increments:
            decision = self.evaluate_workload(data, region, float(shift))
            scenarios.append(decision)
            
            # Stop if we found a safe scenario
            if decision.is_approved:
                break
        
        return scenarios
    
    def find_minimum_shift(
        self,
        data: pd.DataFrame,
        region: str,
    ) -> Optional[float]:
        """
        Find the minimum workload shift needed for approval.
        
        Uses binary search for efficiency.
        
        Args:
            data: DataFrame with telemetry data.
            region: Region to evaluate.
        
        Returns:
            Minimum shift percentage needed, or None if impossible.
        """
        # First check if 100% shift is sufficient
        max_decision = self.evaluate_workload(data, region, 100.0)
        if max_decision.is_blocked:
            return None  # Even 100% shift is not enough
        
        # Check if no shift is needed
        baseline = self.evaluate_workload(data, region, 0.0)
        if baseline.is_approved:
            return 0.0
        
        # Binary search for minimum shift
        low, high = 0.0, 100.0
        while high - low > 1.0:
            mid = (low + high) / 2
            decision = self.evaluate_workload(data, region, mid)
            if decision.is_approved:
                high = mid
            else:
                low = mid
        
        return high
