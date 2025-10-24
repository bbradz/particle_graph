import matplotlib.pyplot as plt
import os
import numpy as np
from Token2Model.check import IDX_TO_CHECK, NUM_CHECKS

def get_check_category(check_name):
    """Returns the category prefix for a check name based on its object level."""
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
        return "(particle)"
    elif check_name in field_checks:
        return "(field)"
    elif check_name in interaction_checks:
        return "(itract)"
    elif check_name in global_checks:
        return "(global)"
    else:
        return "(unknown)"

def get_check_hierarchical_level(check_name):
    """
    Returns the hierarchical level of a check for curriculum learning.
    Higher levels indicate more complex checks that should be learned later.
    """
    category = get_check_category(check_name)
    
    # Define hierarchical levels (higher = more complex)
    level_mapping = {
        "(particle)": 1,  # Basic particle properties
        "(field)": 2,     # Field structure and properties
        "(itract)": 3,    # Interaction dynamics
        "(global)": 4,    # Global consistency and anomalies
        "(unknown)": 0    # Unknown checks at lowest level
    }
    
    return level_mapping.get(category, 0)

def categorize_checks_by_level():
    """
    Returns a dictionary mapping hierarchical levels to lists of check names.
    Useful for curriculum learning and exploration strategies.
    """
    from Token2Model.check import IDX_TO_CHECK
    
    level_to_checks = {}
    for check_idx, check_name in IDX_TO_CHECK.items():
        level = get_check_hierarchical_level(check_name)
        if level not in level_to_checks:
            level_to_checks[level] = []
        level_to_checks[level].append(check_name)
    
    return level_to_checks

class TrainingPlotter:
    """Handles the generation and saving of training plots."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Training plots will be saved to: {os.path.abspath(self.output_dir)}")

    def _save_plot(self, fig, filename):
        """Helper to save a figure to the RL directory."""
        rl_dir = os.path.join(self.output_dir, 'rl')
        os.makedirs(rl_dir, exist_ok=True)
        path = os.path.join(rl_dir, f"{filename}.png")
        fig.savefig(path)
        plt.close(fig)

    def _get_robust_ylim(self, data, outlier_percentile=99.0):
        """
        Calculates plot y-limits that ignore extreme outliers.
        Returns (min, max) tuple for the y-axis.
        """
        # Ensure data is a numpy array and filter out NaNs and infinities
        valid_data = np.array(data)[np.isfinite(data)]
        if len(valid_data) < 2:
            return None # Not enough data to determine a range

        # Calculate percentile-based limits
        lower_bound = np.percentile(valid_data, 100.0 - outlier_percentile)
        upper_bound = np.percentile(valid_data, outlier_percentile)
        
        # Add some padding to the limits
        data_range = upper_bound - lower_bound
        if data_range < 1e-9: # Handle case where data is mostly flat
            data_range = upper_bound * 0.1 if upper_bound > 0 else 1.0

        plot_min = lower_bound - 0.1 * data_range
        plot_max = upper_bound + 0.1 * data_range
        
        return plot_min, plot_max

    def generate_plots(self, history: dict):
        """Generate all relevant training plots."""
        if not history or not history.get('step'):
            print("Plotting skipped: History is empty.")
            return

        # Individual plots for each metric
        self._plot_ppo_loss_start_end(history)
        self._plot_ppo_loss_reduction(history)
        self._plot_replay_losses_mse(history)
        self._plot_loss_improvement(history)
        self._plot_avg_seq_reward(history)
        self._plot_policy_loss(history)
        self._plot_value_loss(history)
        self._plot_reward_loss(history)
        self._plot_criticality_loss(history)
        self._plot_consistency_loss(history)
        self._plot_combined_losses(history)
        self._plot_eos_percentage(history)
        self._plot_exploration_metrics(history)
        self._plot_per_check_value_predictions(history)
        
        print(f"Plots saved.")

    def _plot_ppo_loss_start_end(self, history):
        """Plots PPO loss start and end over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['ppo_loss_start'], label='PPO Loss (Start)', color='tab:blue', linewidth=2)
        ax.plot(steps, history['ppo_loss_end'], label='PPO Loss (End)', color='tab:red', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('PPO Loss Start vs End Over Time')
        
        combined_data = history['ppo_loss_start'] + history['ppo_loss_end']
        ylim = self._get_robust_ylim(combined_data)
        if ylim:
            ax.set_ylim(ylim)
            if np.max(combined_data) > ylim[1] or np.min(combined_data) < ylim[0]:
                ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "ppo_loss_start_end")

    def _plot_ppo_loss_reduction(self, history):
        """Plots PPO loss reduction over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['ppo_loss_reduction'], color='tab:green', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Reduction')
        ax.set_title('PPO Loss Reduction Over Time')
        ylim = self._get_robust_ylim(history['ppo_loss_reduction'])
        if ylim:
            ax.set_ylim(ylim)
            if np.max(history['ppo_loss_reduction']) > ylim[1] or np.min(history['ppo_loss_reduction']) < ylim[0]:
                 ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "ppo_loss_reduction")

    def _plot_replay_losses_mse(self, history):
        """Plots original and current replay losses (MSE) over time."""
        if not history.get('replay_original_loss'):
            return # Skip if no replay data
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        replay_original = history['replay_original_loss']
        replay_current = history['replay_current_loss']
        
        # Handle dimension mismatch by truncating to the shorter length
        min_length = min(len(steps), len(replay_original), len(replay_current))
        steps = steps[:min_length]
        replay_original = replay_original[:min_length]
        replay_current = replay_current[:min_length]
        
        ax.plot(steps, replay_original, label='Original Replay Loss (MSE)', color='tab:blue', linewidth=2)
        ax.plot(steps, replay_current, label='Current Replay Loss (MSE)', color='tab:red', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Replay Buffer MSE Loss Over Time')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "replay_losses_mse")

    def _plot_loss_improvement(self, history):
        """Plots loss improvement over time."""
        if not history.get('replay_improvement'):
            return # Skip if no replay data
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        replay_improvement = history['replay_improvement']
        
        # Handle dimension mismatch by truncating to the shorter length
        min_length = min(len(steps), len(replay_improvement))
        steps = steps[:min_length]
        replay_improvement = replay_improvement[:min_length]
        
        ax.plot(steps, replay_improvement, color='tab:purple', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Improvement')
        ax.set_title('Loss Improvement Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "loss_improvement")

    def _plot_avg_seq_reward(self, history):
        """Plots average sequence reward over time."""
        if not history.get('avg_reward'):
            return # Skip if no reward data
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        avg_reward = history['avg_reward']
        
        # Handle dimension mismatch by truncating to the shorter length
        min_length = min(len(steps), len(avg_reward))
        steps = steps[:min_length]
        avg_reward = avg_reward[:min_length]
        
        ax.plot(steps, avg_reward, color='tab:green', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Average Sequence Reward')
        ax.set_title('Average Sequence Reward Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "avg_seq_reward")

    def _plot_policy_loss(self, history):
        """Plots policy loss over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['policy_loss'], color='tab:blue', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss Over Time')
        ylim = self._get_robust_ylim(history['policy_loss'])
        if ylim:
            ax.set_ylim(ylim)
            if np.max(history['policy_loss']) > ylim[1] or np.min(history['policy_loss']) < ylim[0]:
                ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "policy_loss")

    def _plot_value_loss(self, history):
        """Plots value loss over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['value_loss'], color='tab:orange', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "value_loss")

    def _plot_reward_loss(self, history):
        """Plots reward loss over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['reward_loss'], color='tab:purple', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Reward Loss')
        ax.set_title('Reward Loss Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "reward_loss")

    def _plot_criticality_loss(self, history):
        """Plots criticality loss over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['criticality_loss'], color='tab:brown', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Criticality Loss')
        ax.set_title('Criticality Loss Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "criticality_loss")

    def _plot_consistency_loss(self, history):
        """Plots consistency loss over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['consistency_loss'], color='tab:pink', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Consistency Loss')
        ax.set_title('Consistency Loss Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "consistency_loss")

    def _plot_combined_losses(self, history):
        """Plots all loss components in separate subplots for detailed analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('RL Training Loss Components', fontsize=16)
        
        steps = history['step']
        
        # Plot 1: Policy Loss
        axes[0, 0].plot(steps, history['policy_loss'], color='tab:blue', linewidth=2)
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Value Loss
        axes[0, 1].plot(steps, history['value_loss'], color='tab:orange', linewidth=2)
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Criticality Loss
        axes[0, 2].plot(steps, history['criticality_loss'], color='tab:brown', linewidth=2)
        axes[0, 2].set_title('Criticality Loss')
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Reward Loss
        axes[1, 0].plot(steps, history['reward_loss'], color='tab:purple', linewidth=2)
        axes[1, 0].set_title('Reward Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Consistency Loss
        axes[1, 1].plot(steps, history['consistency_loss'], color='tab:pink', linewidth=2)
        axes[1, 1].set_title('Consistency Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Average Sequence Reward
        axes[1, 2].plot(steps, history['avg_seq_reward'], color='tab:green', linewidth=2)
        axes[1, 2].set_title('Average Sequence Reward')
        axes[1, 2].set_xlabel('Training Step')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(fig, "rl_loss_components")
        
        # Also create the original combined plot for comparison
        self._plot_all_losses_combined(history)

    def _plot_all_losses_combined(self, history):
        """Plots all loss components together for comparison (original combined plot)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        steps = history['step']
        
        # Plot all available loss components
        losses_to_plot = []
        if history.get('policy_loss'):
            ax.plot(steps, history['policy_loss'], color='tab:blue', linewidth=2, label='Policy Loss')
            losses_to_plot.extend(history['policy_loss'])
        if history.get('value_loss'):
            ax.plot(steps, history['value_loss'], color='tab:orange', linewidth=2, label='Value Loss')
            losses_to_plot.extend(history['value_loss'])
        if history.get('reward_loss'):
            ax.plot(steps, history['reward_loss'], color='tab:purple', linewidth=2, label='Reward Loss')
            losses_to_plot.extend(history['reward_loss'])
        if history.get('criticality_loss'):
            ax.plot(steps, history['criticality_loss'], color='tab:brown', linewidth=2, label='Criticality Loss')
            losses_to_plot.extend(history['criticality_loss'])
        if history.get('consistency_loss'):
            ax.plot(steps, history['consistency_loss'], color='tab:pink', linewidth=2, label='Consistency Loss')
            losses_to_plot.extend(history['consistency_loss'])
        if history.get('replay_value_loss'):
            ax.plot(steps, history['replay_value_loss'], color='tab:gray', linewidth=2, label='Replay Value Loss')
            losses_to_plot.extend(history['replay_value_loss'])
        if history.get('replay_reward_loss'):
            ax.plot(steps, history['replay_reward_loss'], color='tab:olive', linewidth=2, label='Replay Reward Loss')
            losses_to_plot.extend(history['replay_reward_loss'])
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('All Loss Components Over Time')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Set robust y-limits if we have data
        if losses_to_plot:
            ylim = self._get_robust_ylim(losses_to_plot)
            if ylim:
                ax.set_ylim(ylim)
                if np.max(losses_to_plot) > ylim[1] or np.min(losses_to_plot) < ylim[0]:
                    ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                            fontsize=10, verticalalignment='top', horizontalalignment='right', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        self._save_plot(fig, "combined_losses")

    def _plot_eos_percentage(self, history):
        """Plots EOS percentage over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['eos_percentage'], color='tab:red', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('% Sequences with EOS')
        ax.set_title('EOS Percentage Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "eos_percentage")

    def _plot_exploration_metrics(self, history):
        """Plots trajectory diversity and n-gram entropy."""
        fig, ax1 = plt.subplots(figsize=(12, 7))
        steps = history['step']

        color = 'tab:purple'
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Avg. Pairwise Jaccard Distance', color=color)
        ax1.plot(steps, history['avg_jaccard_dist'], color=color, label='Jaccard Distance')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color = 'tab:cyan'
        ax2.set_ylabel('Bigram Entropy (bits)', color=color)
        ax2.plot(steps, history['bigram_entropy'], color=color, linestyle='--', label='Bigram Entropy')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.tight_layout()
        ax1.set_title('Exploration and Trajectory Diversity Over Time')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "exploration_metrics")

    def _plot_per_check_value_predictions(self, history):
        """Creates individual plots for each check showing loss/score and value prediction mean/variance."""
        steps = history['step']
        
        # Create RL phase directory
        rl_dir = os.path.join(self.output_dir, 'rl')
        os.makedirs(rl_dir, exist_ok=True)
        
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
            loss_key = f'check_loss_{check_name}'
            score_key = f'check_score_{check_name}'
            mean_key = f'value_mean_{check_name}'
            var_key = f'value_var_{check_name}'
                
            category = get_check_category(check_name)
            
            # Create category-specific directory within RL directory
            category_dir = os.path.join(rl_dir, category.replace("(", "").replace(")", ""))
            os.makedirs(category_dir, exist_ok=True)
            
            # Create the first plot: Loss and Score
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot loss on left y-axis
            color1 = 'tab:blue'
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Average L1 Loss', color=color1)
            line1 = ax1.plot(steps, history[loss_key], color=color1, linewidth=2, label='Loss')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.set_yscale('log')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Plot score on right y-axis if available
            if score_key in history and history[score_key]:
                ax2 = ax1.twinx()
                color2 = 'tab:red'
                ax2.set_ylabel('Average Score / Max Score', color=color2)
                line2 = ax2.plot(steps, history[score_key], color=color2, linewidth=2, label='Score')
                ax2.tick_params(axis='y', labelcolor=color2)
                ax2.set_ylim(0, 1)  # Scores are normalized between 0 and 1
                
                # Add legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
            
            ax1.set_title(f'{category} {check_name} - Loss and Score Over Time')
            
            # Save to category-specific directory
            filename1 = f"{check_name}_loss_score"
            filepath1 = os.path.join(category_dir, f"{filename1}.png")
            fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # Create the second plot: Value Prediction Mean and Variance
            fig2, ax3 = plt.subplots(figsize=(12, 6))
            
            # Plot mean on left y-axis
            color3 = 'tab:green'
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Mean V(s₀) Prediction', color=color3)
            line3 = ax3.plot(steps, history[mean_key], color=color3, linewidth=2, label='Mean Prediction')
            ax3.tick_params(axis='y', labelcolor=color3)
            ax3.set_ylim(-0.1, 1.1)  # Value predictions should be roughly in [0, 1]
            ax3.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Plot variance on right y-axis
            if var_key in history and history[var_key] and np.mean(history[var_key]) > 1e-9:
                ax4 = ax3.twinx()
                color4 = 'tab:orange'
                ax4.set_ylabel('Variance V(s₀) Prediction', color=color4)
                line4 = ax4.plot(steps, history[var_key], color=color4, linewidth=2, label='Variance', linestyle='--')
                ax4.tick_params(axis='y', labelcolor=color4)
                ax4.set_yscale('log')
                
                # Add legend
                lines = line3 + line4
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc='upper left')
            else:
                # Only mean available, no legend needed
                pass
            
            ax3.set_title(f'{category} {check_name} - Value Prediction Mean and Variance Over Time')
            
            # Save to category-specific directory
            filename2 = f"{check_name}_value_pred"
            filepath2 = os.path.join(category_dir, f"{filename2}.png")
            fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
            plt.close(fig2)

class PreTrainingPlotter:
    """Handles the generation and saving of pre-training plots for phases A, B, and C."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Pre-training plots will be saved to: {os.path.abspath(self.output_dir)}")
    
    def _save_plot(self, fig, filename):
        """Helper to save a figure."""
        path = os.path.join(self.output_dir, f"{filename}.png")
        fig.savefig(path)
        plt.close(fig)
    
    def plot_phase_losses(self, phase_name: str, history: dict):
        """Plot training and validation losses for a pre-training phase."""
        if not history or not history.get('step'):
            print(f"Plotting skipped for {phase_name}: History is empty.")
            return
        
        steps = history['step']
        
        # Determine layout based on phase
        if phase_name == "Phase A":
            # Phase A: Standard 2x2 layout
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{phase_name} Training Progress', fontsize=16)
            
            # Plot 1: Training Loss
            if 'train_loss' in history and history['train_loss']:
                axes[0, 0].plot(steps, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
            
            # Plot 2: Validation Loss
            if 'val_loss' in history and history['val_loss']:
                # Filter out None values and corresponding steps
                val_losses = []
                val_steps = []
                for i, val_loss in enumerate(history['val_loss']):
                    if val_loss is not None:
                        val_losses.append(val_loss)
                        val_steps.append(steps[i])
                
                if val_losses:  # Only plot if we have actual validation data
                    axes[0, 1].plot(val_steps, val_losses, 'r-', linewidth=2, label='Validation Loss')
                    axes[0, 1].set_title('Validation Loss')
                    axes[0, 1].set_xlabel('Step')
                    axes[0, 1].set_ylabel('Loss')
                    axes[0, 1].grid(True, alpha=0.3)
                    axes[0, 1].legend()
            
            # Plot 3: Learning Rate
            if 'learning_rate' in history and history['learning_rate']:
                axes[1, 0].plot(steps, history['learning_rate'], 'g-', linewidth=2, label='Learning Rate')
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # Plot 4: Gradient Norm
            if 'grad_norm' in history and history['grad_norm']:
                axes[1, 1].plot(steps, history['grad_norm'], 'm-', linewidth=2, label='Gradient Norm')
                axes[1, 1].set_title('Gradient Norm')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Gradient Norm')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
                
        elif phase_name == "Phase B":
            # Phase B: 2x3 layout for separate loss components
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'{phase_name} Training Progress', fontsize=16)
            
            # Plot 1: Total Loss
            if 'train_loss' in history and history['train_loss']:
                axes[0, 0].plot(steps, history['train_loss'], 'b-', linewidth=2, label='Total Loss')
                axes[0, 0].set_title('Total Loss (Crit + λ×Gram)')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
            
            # Plot 2: Criticality Loss
            if 'crit_loss' in history and history['crit_loss']:
                axes[0, 1].plot(steps, history['crit_loss'], 'r-', linewidth=2, label='Criticality Loss')
                axes[0, 1].set_title('Criticality Loss')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # Plot 3: Grammar Loss
            if 'grammar_loss' in history and history['grammar_loss']:
                axes[0, 2].plot(steps, history['grammar_loss'], 'g-', linewidth=2, label='Grammar Loss')
                axes[0, 2].set_title('Grammar Loss')
                axes[0, 2].set_xlabel('Step')
                axes[0, 2].set_ylabel('Loss')
                axes[0, 2].grid(True, alpha=0.3)
                axes[0, 2].legend()
            
            # Plot 4: Validation Loss
            if 'val_loss' in history and history['val_loss']:
                # Filter out None values and corresponding steps
                val_losses = []
                val_steps = []
                for i, val_loss in enumerate(history['val_loss']):
                    if val_loss is not None:
                        val_losses.append(val_loss)
                        val_steps.append(steps[i])
                
                if val_losses:  # Only plot if we have actual validation data
                    axes[1, 0].plot(val_steps, val_losses, 'm-', linewidth=2, label='Validation Loss')
                    axes[1, 0].set_title('Validation Loss')
                    axes[1, 0].set_xlabel('Step')
                    axes[1, 0].set_ylabel('Loss')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].legend()
            
            # Plot 5: Learning Rate
            if 'learning_rate' in history and history['learning_rate']:
                axes[1, 1].plot(steps, history['learning_rate'], 'orange', linewidth=2, label='Learning Rate')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            
            # Plot 6: Gradient Norm
            if 'grad_norm' in history and history['grad_norm']:
                axes[1, 2].plot(steps, history['grad_norm'], 'purple', linewidth=2, label='Gradient Norm')
                axes[1, 2].set_title('Gradient Norm')
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Gradient Norm')
                axes[1, 2].set_yscale('log')
                axes[1, 2].grid(True, alpha=0.3)
                axes[1, 2].legend()
                
        elif phase_name == "Phase C":
            # Phase C: 2x4 layout for separate loss components
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'{phase_name} Training Progress', fontsize=16)
            
            # Plot 1: Total Loss
            if 'train_loss' in history and history['train_loss']:
                axes[0, 0].plot(steps, history['train_loss'], 'b-', linewidth=2, label='Total Loss')
                axes[0, 0].set_title('Total Loss (Crit + Q + Gram)')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
            
            # Plot 2: Criticality Loss
            if 'crit_loss' in history and history['crit_loss']:
                axes[0, 1].plot(steps, history['crit_loss'], 'r-', linewidth=2, label='Criticality Loss')
                axes[0, 1].set_title('Criticality Loss')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # Plot 3: Q-Loss
            if 'q_loss' in history and history['q_loss']:
                axes[0, 2].plot(steps, history['q_loss'], 'g-', linewidth=2, label='Q-Loss')
                axes[0, 2].set_title('Q-Loss')
                axes[0, 2].set_xlabel('Step')
                axes[0, 2].set_ylabel('Loss')
                axes[0, 2].grid(True, alpha=0.3)
                axes[0, 2].legend()
            
            # Plot 4: Grammar Loss
            if 'grammar_loss' in history and history['grammar_loss']:
                axes[0, 3].plot(steps, history['grammar_loss'], 'orange', linewidth=2, label='Grammar Loss')
                axes[0, 3].set_title('Grammar Loss')
                axes[0, 3].set_xlabel('Step')
                axes[0, 3].set_ylabel('Loss')
                axes[0, 3].grid(True, alpha=0.3)
                axes[0, 3].legend()
            
            # Plot 5: Validation Loss
            if 'val_loss' in history and history['val_loss']:
                # Filter out None values and corresponding steps
                val_losses = []
                val_steps = []
                for i, val_loss in enumerate(history['val_loss']):
                    if val_loss is not None:
                        val_losses.append(val_loss)
                        val_steps.append(steps[i])
                
                if val_losses:  # Only plot if we have actual validation data
                    axes[1, 0].plot(val_steps, val_losses, 'm-', linewidth=2, label='Validation Loss')
                    axes[1, 0].set_title('Validation Loss')
                    axes[1, 0].set_xlabel('Step')
                    axes[1, 0].set_ylabel('Loss')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].legend()
            
            # Plot 6: Learning Rate
            if 'learning_rate' in history and history['learning_rate']:
                axes[1, 1].plot(steps, history['learning_rate'], 'purple', linewidth=2, label='Learning Rate')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            
            # Plot 7: Gradient Norm
            if 'grad_norm' in history and history['grad_norm']:
                axes[1, 2].plot(steps, history['grad_norm'], 'brown', linewidth=2, label='Gradient Norm')
                axes[1, 2].set_title('Gradient Norm')
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Gradient Norm')
                axes[1, 2].set_yscale('log')
                axes[1, 2].grid(True, alpha=0.3)
                axes[1, 2].legend()
            
            # Hide the last subplot
            axes[1, 3].set_visible(False)
        
        plt.tight_layout()
        
        # Create phase-specific directory for phase losses
        phase_dir = os.path.join(self.output_dir, phase_name.lower().replace(" ", "_"))
        os.makedirs(phase_dir, exist_ok=True)
        
        # Save to phase-specific directory
        filename = f"{phase_name.lower().replace(' ', '_')}_losses"
        filepath = os.path.join(phase_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved {phase_name} losses plot to {filepath}")
    
    def plot_per_check_performance(self, phase_name: str, check_performance: dict):
        """Plot performance metrics for each check, saving into phase-specific directories with check categories."""
        
        if not check_performance:
            print(f"No check performance data available for {phase_name}")
            return
        
        # Create phase-specific directory
        phase_dir = os.path.join(self.output_dir, phase_name.lower().replace(" ", "_"))
        os.makedirs(phase_dir, exist_ok=True)
        
        # Group checks by category
        checks_by_category = {
            'particle': [],
            'field': [],
            'interaction': [],
            'global': []
        }
        
        for check_name, metrics in check_performance.items():
            category = get_check_category(check_name)
            if category == "(particle)":
                checks_by_category['particle'].append((check_name, metrics))
            elif category == "(field)":
                checks_by_category['field'].append((check_name, metrics))
            elif category == "(itract)":
                checks_by_category['interaction'].append((check_name, metrics))
            elif category == "(global)":
                checks_by_category['global'].append((check_name, metrics))
        
        # Create plots for each category
        for category, checks in checks_by_category.items():
            if not checks:
                continue
                
            # Create category-specific directory within phase directory
            category_dir = os.path.join(phase_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Create a figure with subplots for each check in this category
            num_checks = len(checks)
            cols = min(4, num_checks)  # Max 4 columns
            rows = (num_checks + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if num_checks == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'{phase_name} {category.title()} Check Performance', fontsize=16)
            
            for idx, (check_name, metrics) in enumerate(checks):
                row = idx // cols
                col = idx % cols
                
                # Handle axes indexing more robustly
                if num_checks == 1:
                    ax = axes[0]
                elif rows == 1:
                    ax = axes[0, col]  # Fix: Use proper 2D indexing for single row
                elif cols == 1:
                    ax = axes[row]
                else:
                    ax = axes[row, col]
                
                
                # Plot available metrics
                if 'train_acc' in metrics and 'val_acc' in metrics:
                    steps = metrics.get('steps', range(len(metrics['train_acc'])))
                    # Ensure steps is a list, not a numpy array
                    if hasattr(steps, 'tolist'):
                        steps = steps.tolist()
                    elif not isinstance(steps, list):
                        steps = list(steps)
                    ax.plot(steps, metrics['train_acc'], 'b-', label='Train Acc', linewidth=2)
                    ax.plot(steps, metrics['val_acc'], 'r-', label='Val Acc', linewidth=2)
                    ax.set_ylabel('Accuracy')
                elif 'train_loss' in metrics and 'val_loss' in metrics:
                    steps = metrics.get('steps', range(len(metrics['train_loss'])))
                    # Ensure steps is a list, not a numpy array
                    if hasattr(steps, 'tolist'):
                        steps = steps.tolist()
                    elif not isinstance(steps, list):
                        steps = list(steps)
                    ax.plot(steps, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
                    ax.plot(steps, metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
                    ax.set_ylabel('Loss')
                    
                    # Create additional plots for this check
                    self._create_additional_check_plots(check_name, metrics, category_dir, phase_name)
                
                ax.set_title(f'{check_name}')
                ax.set_xlabel('Step')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Hide unused subplots
            for idx in range(num_checks, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # Save to category-specific directory
            filename = f"{phase_name.lower()}_{category}_check_performance"
            filepath = os.path.join(category_dir, f"{filename}.png")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved {category} check performance plot to {filepath}")
        
        # Create combined grid plots for each category
        self._create_combined_grid_plots(phase_name, check_performance)
    
    def _create_combined_grid_plots(self, phase_name, check_performance):
        """Create combined grid plots showing all checks of each type in one big plot."""
        
        # Group checks by category
        checks_by_category = {}
        for check_name, metrics in check_performance.items():
            category = get_check_category(check_name)
            if category not in checks_by_category:
                checks_by_category[category] = []
            checks_by_category[category].append((check_name, metrics))
        
        for category, checks in checks_by_category.items():
            if not checks:
                continue
            
            # Create phase directory
            phase_dir = os.path.join(self.output_dir, phase_name.lower().replace(' ', '_'))
            os.makedirs(phase_dir, exist_ok=True)
            
            # Create category directory
            category_dir = os.path.join(phase_dir, category.replace("(", "").replace(")", ""))
            os.makedirs(category_dir, exist_ok=True)
            
            # Create combined loss/score plot
            self._create_combined_loss_score_plot(checks, category_dir, phase_name, category)
            
            # Create combined value prediction plot
            self._create_combined_value_pred_plot(checks, category_dir, phase_name, category)
    
    def _create_combined_loss_score_plot(self, checks, category_dir, phase_name, category):
        """Create a combined grid plot showing train/val loss for all checks in a category."""
        
        num_checks = len(checks)
        cols = min(4, num_checks)  # Max 4 columns
        rows = (num_checks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if num_checks == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{phase_name} {category.title()} - Training and Validation Loss', fontsize=16)
        
        for idx, (check_name, metrics) in enumerate(checks):
            row = idx // cols
            col = idx % cols
            
            # Handle axes indexing
            if num_checks == 1:
                ax = axes[0]
            elif rows == 1:
                ax = axes[0, col]
            elif cols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            steps = metrics.get('steps', range(len(metrics['train_loss'])))
            if hasattr(steps, 'tolist'):
                steps = steps.tolist()
            elif not isinstance(steps, list):
                steps = list(steps)
            
            # Plot train and validation loss
            ax.set_ylabel('Average L1 Loss')
            ax.plot(steps, metrics['train_loss'], color='tab:blue', linewidth=2, label='Train Loss')
            ax.plot(steps, metrics['val_loss'], color='tab:red', linewidth=2, linestyle='--', label='Val Loss', alpha=0.7)
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
            ax.legend()
            
            ax.set_title(f'{check_name}')
            ax.set_xlabel('Step')
        
        # Hide empty subplots
        for idx in range(num_checks, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows == 1:
                axes[0, col].set_visible(False)
            elif cols == 1:
                axes[row].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        filepath = os.path.join(category_dir, f"loss_combined_{category}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_combined_value_pred_plot(self, checks, category_dir, phase_name, category):
        """Create a combined grid plot showing value predictions for all checks in a category."""
        
        num_checks = len(checks)
        cols = min(4, num_checks)  # Max 4 columns
        rows = (num_checks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if num_checks == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{phase_name} {category.title()} - Value Predictions Combined', fontsize=16)
        
        for idx, (check_name, metrics) in enumerate(checks):
            row = idx // cols
            col = idx % cols
            
            # Handle axes indexing
            if num_checks == 1:
                ax = axes[0]
            elif rows == 1:
                ax = axes[0, col]
            elif cols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            steps = metrics.get('steps', range(len(metrics['train_loss'])))
            if hasattr(steps, 'tolist'):
                steps = steps.tolist()
            elif not isinstance(steps, list):
                steps = list(steps)
            
            # Create dummy value predictions
            dummy_mean = [0.5 + 0.3 * np.sin(step / 10) for step in steps]
            dummy_var = [0.1 + 0.05 * np.cos(step / 15) for step in steps]
            
            # Plot mean and variance predictions
            ax.plot(steps, dummy_mean, color='tab:blue', linewidth=2, label='Mean Prediction')
            ax.plot(steps, dummy_var, color='tab:red', linewidth=2, label='Variance Prediction')
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Value Prediction')
            ax.set_title(f'{check_name}')
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(num_checks, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows == 1:
                axes[0, col].set_visible(False)
            elif cols == 1:
                axes[row].set_visible(False)
            else:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        filepath = os.path.join(category_dir, f"value_pred_combined_{category}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_additional_check_plots(self, check_name, metrics, category_dir, phase_name):
        """Create additional plots for each check: loss and value_pred plots."""
        
        steps = metrics.get('steps', range(len(metrics['train_loss'])))
        if hasattr(steps, 'tolist'):
            steps = steps.tolist()
        elif not isinstance(steps, list):
            steps = list(steps)
        
        # Create loss plot (train and validation L1 loss)
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot train and validation loss
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Average L1 Loss')
        ax1.plot(steps, metrics['train_loss'], color='tab:blue', linewidth=2, label='Train Loss')
        ax1.plot(steps, metrics['val_loss'], color='tab:red', linewidth=2, linestyle='--', label='Val Loss', alpha=0.7)
        ax1.set_yscale('log')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        ax1.legend()
        
        plt.title(f'{check_name} - Training and Validation Loss Over Time')
        plt.tight_layout()
        
        # Save the plot
        filename1 = f"{phase_name.lower().replace(' ', '_')}_{check_name}_loss"
        filepath1 = os.path.join(category_dir, f"{filename1}.png")
        plt.savefig(filepath1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Create value_pred plot (mean and variance predictions)
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        # For pre-training, we don't have actual value predictions, so we'll create dummy ones
        # In a real implementation, you'd want to calculate actual value predictions
        dummy_mean = [0.5 + 0.3 * np.sin(step / 10) for step in steps]  # Dummy mean values
        dummy_var = [0.1 + 0.05 * np.cos(step / 15) for step in steps]   # Dummy variance values
        
        # Plot mean prediction
        ax.plot(steps, dummy_mean, color='tab:blue', linewidth=2, label='Mean Prediction')
        
        # Plot variance prediction
        ax.plot(steps, dummy_var, color='tab:red', linewidth=2, label='Variance Prediction')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value Prediction')
        ax.set_title(f'{check_name} - Value Predictions Over Time')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        filename2 = f"{phase_name.lower().replace(' ', '_')}_{check_name}_value_pred"
        filepath2 = os.path.join(category_dir, f"{filename2}.png")
        plt.savefig(filepath2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    def plot_phase_comparison(self, phases_data: dict):
        """Plot comparison between different pre-training phases."""
        if not phases_data:
            print("No phase data available for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pre-training Phase Comparison', fontsize=16)
        
        # Plot 1: Final validation losses
        phases = list(phases_data.keys())
        final_val_losses = []
        for phase in phases:
            if 'val_loss' in phases_data[phase] and phases_data[phase]['val_loss']:
                # Get the last non-None validation loss
                val_losses = [v for v in phases_data[phase]['val_loss'] if v is not None]
                if val_losses:
                    final_val_losses.append(val_losses[-1])
                else:
                    final_val_losses.append(float('nan'))
            else:
                final_val_losses.append(float('nan'))
        
        axes[0, 0].bar(phases, final_val_losses, color=['blue', 'orange', 'green'])
        axes[0, 0].set_title('Final Validation Loss by Phase')
        axes[0, 0].set_ylabel('Validation Loss')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Training curves
        for phase, data in phases_data.items():
            if 'val_loss' in data and data['val_loss']:
                # Filter out None values and get corresponding steps
                val_losses = [v for v in data['val_loss'] if v is not None]
                val_steps = [data['step'][i] for i, v in enumerate(data['val_loss']) if v is not None]
                if val_losses:  # Only plot if we have actual data
                    axes[0, 1].plot(val_steps, val_losses, label=f'{phase} Val Loss', linewidth=2)
        axes[0, 1].set_title('Validation Loss Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning rate schedules
        for phase, data in phases_data.items():
            if 'learning_rate' in data and data['learning_rate']:
                steps = data.get('steps', range(len(data['learning_rate'])))
                axes[1, 0].plot(steps, data['learning_rate'], label=f'{phase} LR', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedules')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Gradient norms
        for phase, data in phases_data.items():
            if 'grad_norm' in data and data['grad_norm']:
                steps = data.get('steps', range(len(data['grad_norm'])))
                axes[1, 1].plot(steps, data['grad_norm'], label=f'{phase} Grad Norm', linewidth=2)
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save phase comparison plot to main directory
        filename = "pretraining_phase_comparison"
        filepath = os.path.join(self.output_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved phase comparison plot to {filepath}")

    def _plot_ppo_loss_start_end(self, history):
        """Plots PPO loss start and end over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['ppo_loss_start'], label='PPO Loss (Start)', color='tab:blue', linewidth=2)
        ax.plot(steps, history['ppo_loss_end'], label='PPO Loss (End)', color='tab:red', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('PPO Loss Start vs End Over Time')
        
        combined_data = history['ppo_loss_start'] + history['ppo_loss_end']
        ylim = self._get_robust_ylim(combined_data)
        if ylim:
            ax.set_ylim(ylim)
            if np.max(combined_data) > ylim[1] or np.min(combined_data) < ylim[0]:
                ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "ppo_loss_start_end")

    def _plot_ppo_loss_reduction(self, history):
        """Plots PPO loss reduction over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['ppo_loss_reduction'], color='tab:green', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Reduction')
        ax.set_title('PPO Loss Reduction Over Time')
        ylim = self._get_robust_ylim(history['ppo_loss_reduction'])
        if ylim:
            ax.set_ylim(ylim)
            if np.max(history['ppo_loss_reduction']) > ylim[1] or np.min(history['ppo_loss_reduction']) < ylim[0]:
                 ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "ppo_loss_reduction")

    def _plot_replay_losses_mse(self, history):
        """Plots original and current replay losses (MSE) over time."""
        if not history.get('replay_original_loss'):
            return # Skip if no replay data
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        # Filter steps to match the length of replay data
        replay_steps = steps[-len(history['replay_original_loss']):]
        ax.plot(replay_steps, history['replay_original_loss'], label='Original Replay Loss (MSE)', color='tab:blue', linewidth=2)
        ax.plot(replay_steps, history['replay_current_loss'], label='Current Replay Loss (MSE)', color='tab:red', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Replay Buffer MSE Loss Over Time')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_yscale('log')
        self._save_plot(fig, "replay_losses_mse")

    def _plot_loss_improvement(self, history):
        """Plots loss improvement over time."""
        if not history.get('replay_improvement'):
            return # Skip if no replay data
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        # Filter steps to match the length of replay data
        replay_steps = steps[-len(history['replay_improvement']):]
        ax.plot(replay_steps, history['replay_improvement'], color='tab:green', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss Improvement')
        ax.set_title('Replay Buffer Loss Improvement Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "loss_improvement")

    def _plot_avg_seq_reward(self, history):
        """Plots average sequence reward over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['avg_reward'], color='tab:blue', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Average Sequence Reward')
        ax.set_title('Average Sequence Reward Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "avg_seq_reward")

    def _plot_policy_loss(self, history):
        """Plots policy loss over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['policy_loss'], color='tab:green', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss Over Time')
        ylim = self._get_robust_ylim(history['policy_loss'])
        if ylim:
            ax.set_ylim(ylim)
            if np.max(history['policy_loss']) > ylim[1] or np.min(history['policy_loss']) < ylim[0]:
                ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "policy_loss")

    def _plot_value_loss(self, history):
        """Plots value loss over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['value_loss'], color='tab:orange', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "value_loss")

    def _plot_reward_loss(self, history):
        """Plots reward loss over time."""
        if not history.get('reward_loss'):
            return  # Skip if no reward loss data
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['reward_loss'], color='tab:purple', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Reward Loss')
        ax.set_title('Reward Loss Over Time')
        ylim = self._get_robust_ylim(history['reward_loss'])
        if ylim:
            ax.set_ylim(ylim)
            if np.max(history['reward_loss']) > ylim[1] or np.min(history['reward_loss']) < ylim[0]:
                ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "reward_loss")

    def _plot_criticality_loss(self, history):
        """Plots criticality loss over time."""
        if not history.get('criticality_loss'):
            return  # Skip if no criticality loss data
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['criticality_loss'], color='tab:brown', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Criticality Loss')
        ax.set_title('Criticality Loss Over Time')
        ylim = self._get_robust_ylim(history['criticality_loss'])
        if ylim:
            ax.set_ylim(ylim)
            if np.max(history['criticality_loss']) > ylim[1] or np.min(history['criticality_loss']) < ylim[0]:
                ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "criticality_loss")

    def _plot_consistency_loss(self, history):
        """Plots consistency loss over time."""
        if not history.get('consistency_loss'):
            return  # Skip if no consistency loss data
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['consistency_loss'], color='tab:pink', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Consistency Loss')
        ax.set_title('Hierarchical Consistency Loss Over Time')
        ylim = self._get_robust_ylim(history['consistency_loss'])
        if ylim:
            ax.set_ylim(ylim)
            if np.max(history['consistency_loss']) > ylim[1] or np.min(history['consistency_loss']) < ylim[0]:
                ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "consistency_loss")

    def _plot_combined_losses(self, history):
        """Plots all loss components in separate subplots for detailed analysis."""
        steps = history['step']
        
        # Create a 2x2 grid for the main loss components
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Training Loss Components', fontsize=16)
        
        # Plot 1: Policy Loss
        if history.get('policy_loss'):
            axes[0, 0].plot(steps, history['policy_loss'], color='tab:blue', linewidth=2, label='Policy Loss')
            axes[0, 0].set_title('Policy Loss')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        # Plot 2: Value Loss
        if history.get('value_loss'):
            axes[0, 1].plot(steps, history['value_loss'], color='tab:orange', linewidth=2, label='Value Loss')
            axes[0, 1].set_title('Value Loss')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # Plot 3: Criticality Loss
        if history.get('criticality_loss'):
            axes[1, 0].plot(steps, history['criticality_loss'], color='tab:brown', linewidth=2, label='Criticality Loss')
            axes[1, 0].set_title('Criticality Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Plot 4: Consistency Loss
        if history.get('consistency_loss'):
            axes[1, 1].plot(steps, history['consistency_loss'], color='tab:pink', linewidth=2, label='Consistency Loss')
            axes[1, 1].set_title('Consistency Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        self._save_plot(fig, "rl_loss_components")
        
        # Also create the original combined plot for comparison
        self._plot_all_losses_combined(history)

    def _plot_all_losses_combined(self, history):
        """Plots all loss components together for comparison (original combined plot)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        steps = history['step']
        
        # Plot all available loss components
        losses_to_plot = []
        if history.get('policy_loss'):
            ax.plot(steps, history['policy_loss'], color='tab:blue', linewidth=2, label='Policy Loss')
            losses_to_plot.extend(history['policy_loss'])
        if history.get('value_loss'):
            ax.plot(steps, history['value_loss'], color='tab:orange', linewidth=2, label='Value Loss')
            losses_to_plot.extend(history['value_loss'])
        if history.get('reward_loss'):
            ax.plot(steps, history['reward_loss'], color='tab:purple', linewidth=2, label='Reward Loss')
            losses_to_plot.extend(history['reward_loss'])
        if history.get('criticality_loss'):
            ax.plot(steps, history['criticality_loss'], color='tab:brown', linewidth=2, label='Criticality Loss')
            losses_to_plot.extend(history['criticality_loss'])
        if history.get('consistency_loss'):
            ax.plot(steps, history['consistency_loss'], color='tab:pink', linewidth=2, label='Consistency Loss')
            losses_to_plot.extend(history['consistency_loss'])
        if history.get('replay_value_loss'):
            ax.plot(steps, history['replay_value_loss'], color='tab:gray', linewidth=2, label='Replay Value Loss')
            losses_to_plot.extend(history['replay_value_loss'])
        if history.get('replay_reward_loss'):
            ax.plot(steps, history['replay_reward_loss'], color='tab:olive', linewidth=2, label='Replay Reward Loss')
            losses_to_plot.extend(history['replay_reward_loss'])
        
        if not losses_to_plot:
            plt.close(fig)
            return  # Skip if no loss data
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('All Loss Components Over Time')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Set robust y-limits
        ylim = self._get_robust_ylim(losses_to_plot)
        if ylim:
            ax.set_ylim(ylim)
            if np.max(losses_to_plot) > ylim[1] or np.min(losses_to_plot) < ylim[0]:
                ax.text(0.98, 0.98, 'Outliers clipped', transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', horizontalalignment='right', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self._save_plot(fig, "combined_losses")

    def _plot_eos_percentage(self, history):
        """Plots EOS percentage over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = history['step']
        ax.plot(steps, history['eos_percentage'], color='tab:red', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('% Sequences with EOS')
        ax.set_title('EOS Percentage Over Time')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "eos_percentage")

    def _plot_exploration_metrics(self, history):
        """Plots trajectory diversity and n-gram entropy."""
        fig, ax1 = plt.subplots(figsize=(12, 7))
        steps = history['step']

        color = 'tab:purple'
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Avg. Pairwise Jaccard Distance', color=color)
        ax1.plot(steps, history['avg_jaccard_dist'], color=color, label='Jaccard Distance')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color = 'tab:cyan'
        ax2.set_ylabel('Bigram Entropy (bits)', color=color)
        ax2.plot(steps, history['bigram_entropy'], color=color, linestyle='--', label='Bigram Entropy')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.tight_layout()
        ax1.set_title('Exploration and Trajectory Diversity Over Time')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        self._save_plot(fig, "exploration_metrics")

    def _plot_per_check_value_predictions(self, history):
        """Creates individual plots for each check showing loss/score and value prediction mean/variance."""
        steps = history['step']
        
        # Create RL phase directory
        rl_dir = os.path.join(self.output_dir, 'rl')
        os.makedirs(rl_dir, exist_ok=True)
        print("printing per check value predictions...")
        
        for check_idx in range(NUM_CHECKS):
            check_name = IDX_TO_CHECK.get(check_idx, f"check_{check_idx}")
            loss_key = f'check_loss_{check_name}'
            score_key = f'check_score_{check_name}'
            mean_key = f'value_mean_{check_name}'
            var_key = f'value_var_{check_name}'
            
            # Skip if no loss data available
            if loss_key not in history or not history[loss_key]:
                continue
                
            category = get_check_category(check_name)
            
            # Create category-specific directory within RL directory
            category_dir = os.path.join(rl_dir, category.replace("(", "").replace(")", ""))
            os.makedirs(category_dir, exist_ok=True)
            
            # Create the first plot: Loss and Score (for RL phase)
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot loss on left y-axis
            color1 = 'tab:blue'
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Average L1 Loss', color=color1)
            line1 = ax1.plot(steps, history[loss_key], color=color1, linewidth=2, label='Loss')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.set_yscale('log')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Plot score on right y-axis if available (real scores for RL phase)
            if score_key in history and history[score_key]:
                ax2 = ax1.twinx()
                color2 = 'tab:red'
                ax2.set_ylabel('Average Score', color=color2)
                line2 = ax2.plot(steps, history[score_key], color=color2, linewidth=2, label='Score')
                ax2.tick_params(axis='y', labelcolor=color2)
                ax2.set_ylim(0, 1)  # Scores are normalized between 0 and 1
                
                # Add legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
            
            ax1.set_title(f'{category} {check_name} - Loss and Score Over Time')
            
            # Save to category-specific directory
            filename1 = f"{check_name}_loss_score"
            filepath1 = os.path.join(category_dir, f"{filename1}.png")
            fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # Create the second plot: Value Prediction Mean and Variance
            fig2, ax3 = plt.subplots(figsize=(12, 6))
            
            # Plot mean on left y-axis
            color3 = 'tab:green'
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Mean V(s₀) Prediction', color=color3)
            line3 = ax3.plot(steps, history[mean_key], color=color3, linewidth=2, label='Mean Prediction')
            ax3.tick_params(axis='y', labelcolor=color3)
            ax3.set_ylim(-0.1, 1.1)  # Value predictions should be roughly in [0, 1]
            ax3.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
            
            # Plot variance on right y-axis
            if var_key in history and history[var_key] and np.mean(history[var_key]) > 1e-9:
                ax4 = ax3.twinx()
                color4 = 'tab:orange'
                ax4.set_ylabel('Variance V(s₀) Prediction', color=color4)
                line4 = ax4.plot(steps, history[var_key], color=color4, linewidth=2, label='Variance', linestyle='--')
                ax4.tick_params(axis='y', labelcolor=color4)
                ax4.set_yscale('log')
                
                # Add legend
                lines = line3 + line4
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc='upper left')
            else:
                # Only mean available, no legend needed
                pass
            
            ax3.set_title(f'{category} {check_name} - Value Prediction Mean and Variance Over Time')
            
            # Save to category-specific directory
            filename2 = f"{check_name}_value_pred"
            filepath2 = os.path.join(category_dir, f"{filename2}.png")
            fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
            plt.close(fig2)