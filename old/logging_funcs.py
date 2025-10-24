"""
Logger Class for Experiment Tracking and Visualization

A comprehensive solution for tracking metrics during experiments (e.g., ML training).
Supports scalar and statistical metrics with EMA, CSV logging, TensorBoard integration,
custom Matplotlib plots, and formatted CLI output.

Key Features:
- Define metrics (scalar/stat) with CSV destinations and TensorBoard options
- Log values and aggregate per step with EMA for statistical metrics
- Save to CSV files and optionally to TensorBoard
- Generate publication-ready plots with custom styling
- Print formatted metric summaries to CLI

Example Usage:
logger = Logger(debug=True)
logger.metric("loss", "scalar", dest="train_logs.csv", tb=True)
logger.metric("accuracy", "stat", dest="eval_logs.csv", tb=True)

for step in range(100):
    logger.log("loss", current_loss)
    logger.log("accuracy", current_accuracy)
    logger.print_metrics("loss", "accuracy")  # Optional
    logger.step(dest="train_logs.csv")
    logger.step(dest="eval_logs.csv")

logger.plot("loss", dest="train_logs.csv", title="Training Loss")
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
import collections
import time

class Logger:
    def __init__(self, debug: bool = False):
        self.metrics = {}
        self.log_data_buffer = collections.defaultdict(list)
        self.step_num = 0
        self.tb_writers = {}
        self.debug = debug
        self._apply_plot_style()

    def _apply_plot_style(self):
        """Applies the default Matplotlib style settings."""
        plt.rcParams.update({
            'axes.prop_cycle': plt.cycler('color', ['89b4fa', 'fab387', 'a6e3a1', 'f38ba8', 'cba6f7', 'eba0ac', 'f5c2e7', 'f5e0dc', '94e2d5', 'b4befe']),
            'image.cmap': 'mocha',
            'text.color': 'cdd6f4',
            'axes.labelcolor': 'cdd6f4',
            'xtick.labelcolor': 'cdd6f4',
            'ytick.labelcolor': 'cdd6f4',
            'figure.facecolor': '1e1e2e',
            'axes.facecolor': '1e1e2e',
            'savefig.facecolor': '1e1e2e',
            'axes.edgecolor': '313244',
            'legend.edgecolor': '313244',
            'xtick.color': '313244',
            'ytick.color': '313244',
            'patch.edgecolor': '313244',
            'hatch.color': '313244',
            'grid.color': '313244',
            'boxplot.flierprops.color': 'cdd6f4',
            'boxplot.flierprops.markerfacecolor': 'cdd6f4',
            'boxplot.flierprops.markeredgecolor': 'cdd6f4',
            'boxplot.boxprops.color': 'cdd6f4',
            'boxplot.whiskerprops.color': 'cdd6f4',
            'boxplot.capprops.color': 'cdd6f4',
            'boxplot.medianprops.color': 'cdd6f4',
            'boxplot.meanprops.color': 'cdd6f4',
            'boxplot.meanprops.markerfacecolor': 'cdd6f4',
            'boxplot.meanprops.markeredgecolor': 'cdd6f4',
        })

    def metric(self, name: str, kind: str, dest: str = "logs.csv", tb: bool = False):
        """
        Defines a metric to be tracked by the logger.

        Args:
            name (str): The name of the metric.
            kind (str): The type of metric, either "stat" (for EMA) or "scalar".
            dest (str): The default destination CSV file for this metric.
            tb (bool): Whether to log this metric to TensorBoard.
        """
        if self.debug:
            print(f"[Logger] Defining metric '{name}' (kind: {kind}, dest: {dest}, tb: {tb})")
        if kind not in ["stat", "scalar"]: raise ValueError("Kind must be 'stat' or 'scalar'.")
        self.metrics[name] = {
            "kind": kind,
            "data": [],
            "destinations": {dest},
            "tb": tb
        }
        if kind == "stat":
            self.metrics[name]["ema_mean"] = 0.0
            self.metrics[name]["ema_variance"] = 0.0
            self.metrics[name]["alpha"] = 0.1
            self.metrics[name]["count"] = 0
        if tb and dest not in self.tb_writers:
            log_dir = f"runs/{os.path.splitext(os.path.basename(dest))[0]}"
            self.tb_writers[dest] = SummaryWriter(log_dir=log_dir)

    def log(self, name: str, value: float, dest: str = None):
        """
        Logs a new value for a metric.

        Args:
            name (str): The name of the metric to update.
            value (float): The new value to log.
            dest (str, optional): The specific destination for this log entry. If None,
                                the default destination(s) from metric() will be used.
        """
        if name not in self.metrics:
            raise ValueError(f"Metric '{name}' not defined. Please define it using metric() first.")
        if self.debug:
            destinations_to_log_str = {dest} if dest else self.metrics[name]['destinations']
            print(f"[Logger] Logging to buffer for step {self.step_num}: '{name}' = {value:.4f} for dest(s) {destinations_to_log_str}")

        metric_info = self.metrics[name]
        metric_info["data"].append(value)
        if metric_info["kind"] == "stat":
            alpha = metric_info["alpha"]
            current_mean = metric_info["ema_mean"]
            current_variance = metric_info["ema_variance"]
            metric_info["count"] += 1
            new_mean = alpha * value + (1 - alpha) * current_mean
            metric_info["ema_mean"] = new_mean
            if metric_info["count"] > 1:
                new_variance = alpha * (value - current_mean) * (value - new_mean) + (1 - alpha) * current_variance
                metric_info["ema_variance"] = new_variance
            else:
                metric_info["ema_variance"] = 0.0
        log_entry = {"step": self.step_num, "metric_name": name, "value": value}
        if metric_info["kind"] == "stat":
            log_entry["ema_mean"] = metric_info["ema_mean"]
            log_entry["ema_std"] = metric_info["ema_variance"]**0.5
        destinations_to_log = {dest} if dest else metric_info["destinations"]
        for d in destinations_to_log:
            self.log_data_buffer[d].append(log_entry)

    def step(self, dest):
        """
        Writes all buffered log data for a specific destination to its CSV file
        and flushes it to TensorBoard if configured. This is typically called once per step.

        Args:
            dest (str): The specific destination CSV file to save.
        """
        current_dest = dest
        if self.debug:
            num_entries = len(self.log_data_buffer.get(current_dest, []))
            print(f"[Logger] Step {self.step_num}: Writing {num_entries} buffered entries to '{current_dest}'.")
        if current_dest in self.log_data_buffer and self.log_data_buffer[current_dest]:
            df = pd.DataFrame(self.log_data_buffer[current_dest])
            all_cols = ['step', 'metric_name', 'value', 'ema_mean', 'ema_std']
            df = df.reindex(columns=all_cols)
            if not os.path.exists(current_dest):
                df.to_csv(current_dest, index=False, mode='w', header=True)
            else:
                df.to_csv(current_dest, index=False, mode='a', header=False)
            if current_dest in self.tb_writers:
                writer = self.tb_writers[current_dest]
                latest_entries_for_tb = df.drop_duplicates(subset=['metric_name'], keep='last')
                for _, row in latest_entries_for_tb.iterrows():
                    metric_name = row['metric_name']
                    if metric_name in self.metrics and self.metrics[metric_name].get('tb', False):
                        if self.metrics[metric_name]["kind"] == "scalar":
                            writer.add_scalar(metric_name, row['value'], self.step_num)
                        elif self.metrics[metric_name]["kind"] == "stat":
                            writer.add_scalar(f"{metric_name}/raw", row['value'], self.step_num)
                            if pd.notna(row['ema_mean']):
                                writer.add_scalar(f"{metric_name}/ema_mean", row['ema_mean'], self.step_num)
                            if pd.notna(row['ema_std']):
                                writer.add_scalar(f"{metric_name}/ema_std", row['ema_std'], self.step_num)
                writer.flush()
                # print("Flushed TensorBoard writer for", current_dest)
            self.log_data_buffer[current_dest] = []
        self.step_num += 1


    def print_metrics(self, *metric_names: str):
        """
        Prints the latest values of specified metrics to the console, prefixed with a timestamp,
        with each metric on a new line and horizontally aligned.

        Args:
            *metric_names (str): Variable number of metric names to print.
        """
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        # Determine the maximum length for metric names for alignment
        max_name_len = 0
        for name in metric_names:
            if name in self.metrics:
                max_name_len = max(max_name_len, len(name))
            else: # For not-found metrics, consider the name length + " (not found)"
                max_name_len = max(max_name_len, len(name) + len(": N/A (not found)") - len(": N/A"))


        header = f"[{timestamp} | Step {self.step_num}]:"
        output_lines = [header]

        for name in metric_names:
            line_prefix = f"{' ' * (len(header) - 1)}{' ' * 2}" # Align subsequent lines with the header's content
            
            if name not in self.metrics:
                output_lines.append(f"{line_prefix}{name:<{max_name_len}}: N/A (not found)")
                continue

            metric_info = self.metrics[name]
            if not metric_info["data"]:
                output_lines.append(f"{line_prefix}{name:<{max_name_len}}: No data")
                continue

            latest_value = metric_info["data"][-1]
            if metric_info["kind"] == "scalar":
                output_lines.append(f"{line_prefix}{name:<{max_name_len}}: {latest_value:.4f}")
            elif metric_info["kind"] == "stat":
                ema_mean = metric_info["ema_mean"]
                ema_std = metric_info["ema_variance"]**0.5
                if metric_info["count"] <= 1:
                     output_lines.append(f"{line_prefix}{name:<{max_name_len}}: {latest_value:.4f} (EMA: --.-- +/- --.--)")
                else:
                    output_lines.append(f"{line_prefix}{name:<{max_name_len}}: {latest_value:.4f} (EMA: {ema_mean:.4f} +/- {ema_std:.4f})")
        
        print("\n".join(output_lines))


    def plot(self, metric_name: str, dest: str = "logs.csv", x_axis_name: str = "Step", y_axis_name: str = "Value", title: str = None, plot_style: str = '.-', custom_plot_func=None):
        """
        Generates and displays a plot for a given metric from a CSV log file.

        Args:
            metric_name (str): The name of the metric to plot.
            dest (str): The CSV file from which to read the data for plotting.
            x_axis_name (str): Label for the x-axis.
            y_axis_name (str): Label for the y-axis.
            title (str): The title of the plot. Defaults to "Metric_Name over Steps".
            plot_style (str): Matplotlib plot style (e.g., '.-', '--', 'o').
            custom_plot_func (callable, optional): A function to add custom plot elements.
                                                  It should accept `plt` and `metric_df` as arguments.
        """
        if metric_name not in self.metrics:
            print(f"Error: Metric '{metric_name}' not found. Please ensure it was initialized.")
            return

        try:
            df = pd.read_csv(dest)
            metric_df = df[df['metric_name'] == metric_name].copy()
        except FileNotFoundError:
            print(f"Error: CSV file '{dest}' not found. Please ensure you have saved data to this destination using logger.step().")
            return
        except Exception as e:
            print(f"An error occurred while reading '{dest}': {e}")
            return
        if metric_df.empty:
            print(f"No data available for metric '{metric_name}' in '{dest}' to plot.")
            return

        plt.figure(figsize=(10, 6))
        metric_kind = self.metrics[metric_name]["kind"]
        if metric_kind == "scalar":
            plt.plot(metric_df['step'], metric_df['value'], plot_style, label=metric_name)
        elif metric_kind == "stat":
            plt.plot(metric_df['step'], metric_df['value'], plot_style, label=f'{metric_name} (Raw)')
            if 'ema_mean' in metric_df.columns and 'ema_std' in metric_df.columns:
                plt.plot(metric_df['step'], metric_df['ema_mean'], 'r-', label=f'{metric_name} (EMA Mean)')
                valid_indices = metric_df['ema_std'].notna()
                plt.fill_between(metric_df['step'][valid_indices],
                                 metric_df['ema_mean'][valid_indices] - metric_df['ema_std'][valid_indices],
                                 metric_df['ema_mean'][valid_indices] + metric_df['ema_std'][valid_indices],
                                 color='red', alpha=0.2, label='EMA Std Dev')
            else:
                print(f"Warning: 'ema_mean' or 'ema_std' columns not found in '{dest}' for stat metric '{metric_name}'. Plotting only raw values.")

        # Call custom plot function if provided
        if custom_plot_func:
            try:
                custom_plot_func(plt, metric_df)
            except Exception as e:
                print(f"Error executing custom plot function: {e}")

        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        plt.title(title if title else f"{metric_name} over Steps (from {os.path.basename(dest)})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()