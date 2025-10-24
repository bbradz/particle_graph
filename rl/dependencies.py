from typing import Dict, List, Set, Tuple
from collections import defaultdict
import networkx as nx
from Token2Model.check import IDX_TO_CHECK, NUM_CHECKS

class HierarchicalDependencies:
    """
    Defines and manages hierarchical relationships between check categories.
    Creates dependency graphs used by both exploration and learning logic.
    """
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.check_to_category = {}
        self.category_dependencies = {}
        self._build_dependency_graph()
    
    def _build_dependency_graph(self):
        """Build the hierarchical dependency graph for checks."""
        
        # Define category hierarchy (lower levels depend on higher levels)
        category_hierarchy = {
            "particle": 1,    # Basic particle properties
            "field": 2,       # Field structure (depends on particles)
            "interaction": 3, # Interactions (depends on fields)
            "global": 4       # Global consistency (depends on interactions)
        }
        
        # Map each check to its category
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
            category = self._get_check_category(check_name)
            self.check_to_category[check_name] = category
        
        # Build category dependencies
        self.category_dependencies = {
            "particle": [],  # No dependencies
            "field": ["particle"],  # Fields depend on particles
            "interaction": ["field", "particle"],  # Interactions depend on fields and particles
            "global": ["interaction", "field", "particle"]  # Global checks depend on everything
        }
        
        # Add nodes and edges to the graph
        for category in category_hierarchy:
            self.dependency_graph.add_node(category, level=category_hierarchy[category])
        
        for category, dependencies in self.category_dependencies.items():
            for dep in dependencies:
                self.dependency_graph.add_edge(dep, category)
    
    def _get_check_category(self, check_name: str) -> str:
        """Determine the category of a check based on its name."""
        # Particle Checks
        particle_checks = [
            '_type_check', '_name_check', '_mass_check', '_charge_check'
        ]
        
        # Field Checks
        field_checks = [
            '_name_check', '_type_check', '_groups_check', '_reps_check', 
            '_dim_check', '_gen_check', '_particles_check', '_self_conjugate_check', 
            '_sort_reps', '_reps_dim_consistency', '_gen_type_consistency', 
            '_allowed_charges', '_deplicate_particles', '_particle_numbers', 
            '_particle_types', '_particle_charges', '_all_particle_pass', 
            '_sort_particles', '_chirality_check', '_assign_colors',
            '_mass_term_check', '_potential_term_check'
        ]
        
        # Interaction Checks
        interaction_checks = [
            '_field_length_check', '_field_check', '_params_check', 
            '_all_field_pass_checks', '_check_replicate_fields', '_sort_field',
            '_gen_check', '_dim_check', '_dirac_bilinear_product', 
            '_get_massive_particles', '_check_U1Y_gauge_symmetry', 
            '_yukawa_mass', '_yukawa_matrix', '_mixing_matrix', 
            '_yukawa_lagrangian', '_scalar_mass', '_scalar_quartic', '_scalar_lagrangian'
        ]
        
        # Global Anomaly Checks
        global_checks = [
            '(left)^3', '(color)^3', '(hypercharge)x(left)^2', 
            '(hypercharge)x(color)^2', '(hypercharge)^3', '(hypercharge)^2-grav'
        ]
        
        if check_name in particle_checks:
            return "particle"
        elif check_name in field_checks:
            return "field"
        elif check_name in interaction_checks:
            return "interaction"
        elif check_name in global_checks:
            return "global"
        else:
            return "unknown"
    
    def get_check_dependencies(self, check_name: str) -> List[str]:
        """Get all checks that the given check depends on."""
        category = self.check_to_category.get(check_name, "unknown")
        if category == "unknown":
            return []
        
        dependencies = []
        for dep_category in self.category_dependencies.get(category, []):
            # Find all checks in the dependency category
            for check_idx in range(NUM_CHECKS):
                dep_check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
                if self.check_to_category.get(dep_check_name) == dep_category:
                    dependencies.append(dep_check_name)
        
        return dependencies
    
    def get_category_level(self, category: str) -> int:
        """Get the hierarchical level of a category."""
        return self.dependency_graph.nodes.get(category, {}).get('level', 0)
    
    def get_check_hierarchical_level(self, check_name: str) -> int:
        """Get the hierarchical level of a specific check."""
        category = self.check_to_category.get(check_name, "unknown")
        return self.get_category_level(category)
    
    def get_checks_by_level(self, level: int) -> List[str]:
        """Get all checks at a specific hierarchical level."""
        checks = []
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
            category = self.check_to_category.get(check_name, "unknown")
            if self.get_category_level(category) == level:
                checks.append(check_name)
        return checks
    
    def get_curriculum_phase_checks(self, phase: int) -> List[str]:
        """Get checks that should be active in a specific curriculum phase."""
        if phase == 1:
            return self.get_checks_by_level(1)  # Particle checks
        elif phase == 2:
            return self.get_checks_by_level(1) + self.get_checks_by_level(2)  # Particle + Field
        elif phase == 3:
            return (self.get_checks_by_level(1) + self.get_checks_by_level(2) + 
                   self.get_checks_by_level(3))  # Particle + Field + Interaction
        elif phase == 4:
            return (self.get_checks_by_level(1) + self.get_checks_by_level(2) + 
                   self.get_checks_by_level(3) + self.get_checks_by_level(4))  # All checks
        else:
            return []
    
    def get_exploration_weights(self, current_phase: int) -> Dict[str, float]:
        """
        Get exploration weights for checks based on current curriculum phase.
        Higher weights for checks that should be explored more in the current phase.
        """
        weights = {}
        
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
            category = self.check_to_category.get(check_name, "unknown")
            level = self.get_category_level(category)
            
            if level == 0:  # Unknown checks
                weights[check_name] = 0.1
            elif level <= current_phase:
                # Higher weight for checks at or below current phase
                weights[check_name] = 1.0 / level
            else:
                # Lower weight for checks above current phase
                weights[check_name] = 0.1 / (level - current_phase + 1)
        
        return weights
    
    def get_consistency_loss_weights(self) -> Dict[str, float]:
        """
        Get weights for hierarchical consistency loss.
        Higher weights for checks that have more dependencies.
        """
        weights = {}
        
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
            dependencies = self.get_check_dependencies(check_name)
            # Weight is proportional to number of dependencies
            weights[check_name] = 1.0 + len(dependencies) * 0.1
        
        return weights
    
    def validate_check_consistency(self, check_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Validate that check scores are consistent with hierarchical dependencies.
        Returns a dictionary of consistency violations.
        """
        violations = {}
        
        for check_name, score in check_scores.items():
            dependencies = self.get_check_dependencies(check_name)
            for dep_check in dependencies:
                if dep_check in check_scores:
                    dep_score = check_scores[dep_check]
                    # If dependency fails, this check should also fail
                    if dep_score < 0.5 and score > 0.5:
                        violation_key = f"{check_name}_depends_on_{dep_check}"
                        violations[violation_key] = abs(score - dep_score)
        
        return violations

# Global instance for easy access
HIERARCHICAL_DEPS = HierarchicalDependencies()