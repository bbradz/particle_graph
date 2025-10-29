from typing import Dict, List, Set, Optional
from collections import deque
from dataclasses import dataclass
from config import Config
from .dependencies import HIERARCHICAL_DEPS

@dataclass
class CurriculumPhase:
    """Represents a single phase in the curriculum learning process."""
    phase_id: int
    name: str
    step_range: tuple  # (start_step, end_step)
    active_checks: List[str]
    exploration_weight: float
    temperature_scale: float
    description: str

class CurriculumManager:
    """
    Manages the hierarchical curriculum learning process.
    Orchestrates training phases based on global step count.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.current_phase = 1
        self.phases = self._initialize_phases()
        self.step_count = 0
        # Rolling window history for robust transitions (optional)
        self.success_history = deque(maxlen=config.CURRICULUM_WINDOW_SIZE)
        self.transition_threshold = config.CURRICULUM_TRANSITION_THRESHOLD
        
    def _initialize_phases(self) -> Dict[int, CurriculumPhase]:
        """Initialize the curriculum phases based on configuration."""
        phases = {}
        
        # Phase 1: Basic particle and field checks
        phases[1] = CurriculumPhase(
            phase_id=1,
            name="Basic Structure",
            step_range=(0, self.config.CURRICULUM_PHASE_1_STEPS),
            active_checks=HIERARCHICAL_DEPS.get_curriculum_phase_checks(1),
            exploration_weight=1.0,
            temperature_scale=1.0,
            description="Learn basic particle and field structure"
        )
        
        # Phase 2: Add interaction checks
        phases[2] = CurriculumPhase(
            phase_id=2,
            name="Interactions",
            step_range=(self.config.CURRICULUM_PHASE_1_STEPS, 
                       self.config.CURRICULUM_PHASE_1_STEPS + self.config.CURRICULUM_PHASE_2_STEPS),
            active_checks=HIERARCHICAL_DEPS.get_curriculum_phase_checks(2),
            exploration_weight=0.8,
            temperature_scale=0.9,
            description="Learn interaction dynamics"
        )
        
        # Phase 3: Add global anomaly checks
        phases[3] = CurriculumPhase(
            phase_id=3,
            name="Global Consistency",
            step_range=(self.config.CURRICULUM_PHASE_1_STEPS + self.config.CURRICULUM_PHASE_2_STEPS,
                       self.config.CURRICULUM_PHASE_1_STEPS + self.config.CURRICULUM_PHASE_2_STEPS + self.config.CURRICULUM_PHASE_3_STEPS),
            active_checks=HIERARCHICAL_DEPS.get_curriculum_phase_checks(3),
            exploration_weight=0.6,
            temperature_scale=0.8,
            description="Learn global consistency and anomalies"
        )
        
        # Phase 4: Full complexity
        phases[4] = CurriculumPhase(
            phase_id=4,
            name="Full Complexity",
            step_range=(self.config.CURRICULUM_PHASE_1_STEPS + self.config.CURRICULUM_PHASE_2_STEPS + self.config.CURRICULUM_PHASE_3_STEPS,
                       self.config.CURRICULUM_PHASE_1_STEPS + self.config.CURRICULUM_PHASE_2_STEPS + self.config.CURRICULUM_PHASE_3_STEPS + self.config.CURRICULUM_PHASE_4_STEPS),
            active_checks=HIERARCHICAL_DEPS.get_curriculum_phase_checks(4),
            exploration_weight=0.4,
            temperature_scale=0.7,
            description="Full complexity with all checks"
        )
        
        return phases
    
    def update_step(self, step: int):
        """Update the current step and potentially transition to a new phase."""
        self.step_count = step
        
        # Check if we should transition to a new phase
        for phase_id, phase in self.phases.items():
            if phase.step_range[0] <= step < phase.step_range[1]:
                if self.current_phase != phase_id:
                    self.current_phase = phase_id
                    print(f"Curriculum transition: Entering Phase {phase_id} - {phase.name}")
                    print(f"  Description: {phase.description}")
                    print(f"  Active checks: {len(phase.active_checks)}")
                break
        else:
            # If we're beyond all defined phases, stay in the last phase
            if self.current_phase != 4:
                self.current_phase = 4
                print(f"Curriculum transition: Entering Phase 4 - Full Complexity (beyond defined range)")
    
    def get_current_phase(self) -> CurriculumPhase:
        """Get the current curriculum phase."""
        return self.phases[self.current_phase]
    
    def get_active_checks(self) -> List[str]:
        """Get the list of checks that should be active in the current phase."""
        return self.get_current_phase().active_checks
    
    def get_exploration_weight(self) -> float:
        """Get the exploration weight for the current phase."""
        return self.get_current_phase().exploration_weight
    
    def get_temperature_scale(self) -> float:
        """Get the temperature scaling factor for the current phase."""
        return self.get_current_phase().temperature_scale
    
    def get_phase_progress(self) -> float:
        """Get the progress through the current phase (0.0 to 1.0)."""
        phase = self.get_current_phase()
        phase_start, phase_end = phase.step_range
        phase_length = phase_end - phase_start
        
        if phase_length == 0:
            return 1.0
        
        progress = (self.step_count - phase_start) / phase_length
        return max(0.0, min(1.0, progress))
    
    def should_focus_on_check(self, check_name: str) -> bool:
        """Check if a specific check should be focused on in the current phase."""
        return check_name in self.get_active_checks()
    
    def get_check_priority(self, check_name: str) -> float:
        """Get the priority weight for a specific check in the current phase."""
        if not self.should_focus_on_check(check_name):
            return 0.1  # Low priority for inactive checks
        
        # Higher priority for checks that are newly introduced in this phase
        phase = self.get_current_phase()
        if phase.phase_id == 1:
            return 1.0
        elif phase.phase_id == 2:
            # Higher priority for interaction checks
            if HIERARCHICAL_DEPS.get_check_hierarchical_level(check_name) == 3:
                return 1.0
            else:
                return 0.7
        elif phase.phase_id == 3:
            # Higher priority for global checks
            if HIERARCHICAL_DEPS.get_check_hierarchical_level(check_name) == 4:
                return 1.0
            else:
                return 0.5
        else:
            return 0.3  # Balanced priority in final phase
    
    def get_curriculum_metrics(self) -> Dict[str, float]:
        """Get metrics about the current curriculum state."""
        phase = self.get_current_phase()
        return {
            'current_phase': self.current_phase,
            'phase_progress': self.get_phase_progress(),
            'active_checks_count': len(phase.active_checks),
            'exploration_weight': phase.exploration_weight,
            'temperature_scale': phase.temperature_scale,
            'step_count': self.step_count
        }
    
    def get_phase_summary(self) -> str:
        """Get a human-readable summary of the current phase."""
        phase = self.get_current_phase()
        progress = self.get_phase_progress()
        
        return (f"Phase {phase.phase_id}: {phase.name} "
                f"({progress:.1%} complete)\n"
                f"  Description: {phase.description}\n"
                f"  Active checks: {len(phase.active_checks)}\n"
                f"  Exploration weight: {phase.exploration_weight:.2f}\n"
                f"  Temperature scale: {phase.temperature_scale:.2f}")

    # ================================
    # Rolling window advancement API
    # ================================
    def update_performance(self, success_rate_on_active_checks: float):
        """Record latest success metric for robust phase transition decisions."""
        self.success_history.append(success_rate_on_active_checks)

    def check_and_transition(self):
        """Advance phase if average success in the window exceeds threshold."""
        # If using step-based curriculum only, skip when window not filled
        if self.current_phase >= len(self.phases):
            return
        if len(self.success_history) < self.success_history.maxlen:
            return
        avg_success = sum(self.success_history) / len(self.success_history)
        if avg_success >= self.transition_threshold:
            # Move to next phase if available
            next_phase = min(self.current_phase + 1, max(self.phases.keys()))
            if next_phase != self.current_phase:
                self.current_phase = next_phase
                self.success_history.clear()
                phase = self.get_current_phase()
                print(f"\nCURRICULUM TRANSITION -> {phase.name}\n")

# Legacy Curriculum class for backward compatibility
class Curriculum:
    """Legacy curriculum class for backward compatibility."""
    
    def __init__(self):
        self.manager = None
    
    def initialize(self, config: Config):
        """Initialize the curriculum manager."""
        self.manager = CurriculumManager(config)
    
    def update_step(self, step: int):
        """Update the current step."""
        if self.manager:
            self.manager.update_step(step)
    
    def get_current_phase(self):
        """Get the current phase."""
        if self.manager:
            return self.manager.get_current_phase()
        return None