"""
Metrics collection and reporting for SO100 simulation benchmarks.

Collects performance metrics during evaluation and generates
reports with statistics and visualizations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

from .task_evaluator import TaskResult, TaskStatus


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: int
    task: str
    policy: str
    success: bool
    steps: int
    completion_time: float
    final_distance: Optional[float]
    grasp_achieved: bool
    path_length: float
    path_efficiency: Optional[float]  # optimal / actual path
    timestamp: str


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""
    policy: str
    task: str
    total_episodes: int
    success_count: int
    success_rate: float
    avg_completion_time: Optional[float]
    std_completion_time: Optional[float]
    avg_steps: float
    avg_path_length: float
    grasp_success_rate: float
    avg_final_distance: Optional[float]


class MetricsCollector:
    """
    Collects and aggregates metrics during simulation evaluation.

    Provides:
    - Per-episode metrics recording
    - Aggregated statistics
    - Export to CSV/JSON
    - Report generation
    """

    def __init__(
        self,
        output_dir: str = "results",
        policy_name: str = "unknown",
        task_name: str = "unknown",
    ):
        """
        Initialize the metrics collector.

        Args:
            output_dir: Directory for saving results
            policy_name: Name of the policy being evaluated
            task_name: Name of the task
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.policy_name = policy_name
        self.task_name = task_name

        self.episodes: list[EpisodeMetrics] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def record_episode(
        self,
        episode_id: int,
        result: TaskResult,
        optimal_path_length: Optional[float] = None,
    ) -> EpisodeMetrics:
        """
        Record metrics for a completed episode.

        Args:
            episode_id: Episode number
            result: TaskResult from evaluator
            optimal_path_length: Optimal path length for efficiency calc

        Returns:
            EpisodeMetrics for the episode
        """
        # Calculate path efficiency
        path_efficiency = None
        if optimal_path_length and result.path_length > 0:
            path_efficiency = optimal_path_length / result.path_length

        metrics = EpisodeMetrics(
            episode_id=episode_id,
            task=self.task_name,
            policy=self.policy_name,
            success=result.success,
            steps=result.steps_taken,
            completion_time=result.completion_time,
            final_distance=result.final_distance,
            grasp_achieved=result.grasp_achieved,
            path_length=result.path_length,
            path_efficiency=path_efficiency,
            timestamp=datetime.now().isoformat(),
        )

        self.episodes.append(metrics)
        return metrics

    def get_aggregated_metrics(self) -> BenchmarkMetrics:
        """Calculate aggregated metrics for all recorded episodes."""
        if not self.episodes:
            return BenchmarkMetrics(
                policy=self.policy_name,
                task=self.task_name,
                total_episodes=0,
                success_count=0,
                success_rate=0.0,
                avg_completion_time=None,
                std_completion_time=None,
                avg_steps=0.0,
                avg_path_length=0.0,
                grasp_success_rate=0.0,
                avg_final_distance=None,
            )

        successes = [e for e in self.episodes if e.success]
        success_count = len(successes)
        total = len(self.episodes)

        # Completion time stats (only for successful episodes)
        if successes:
            times = [e.completion_time for e in successes]
            avg_time = np.mean(times)
            std_time = np.std(times)
        else:
            avg_time = None
            std_time = None

        # Final distance (average across all episodes)
        distances = [e.final_distance for e in self.episodes if e.final_distance is not None]
        avg_distance = np.mean(distances) if distances else None

        # Grasp success rate
        grasp_count = sum(1 for e in self.episodes if e.grasp_achieved)

        return BenchmarkMetrics(
            policy=self.policy_name,
            task=self.task_name,
            total_episodes=total,
            success_count=success_count,
            success_rate=success_count / total,
            avg_completion_time=avg_time,
            std_completion_time=std_time,
            avg_steps=np.mean([e.steps for e in self.episodes]),
            avg_path_length=np.mean([e.path_length for e in self.episodes]),
            grasp_success_rate=grasp_count / total,
            avg_final_distance=avg_distance,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert episode metrics to pandas DataFrame."""
        return pd.DataFrame([asdict(e) for e in self.episodes])

    def save_results(self, prefix: str = "") -> dict[str, Path]:
        """
        Save results to files.

        Args:
            prefix: Optional prefix for filenames

        Returns:
            Dict mapping result type to file path
        """
        if prefix:
            prefix = f"{prefix}_"

        base_name = f"{prefix}{self.policy_name}_{self.task_name}_{self.run_id}"
        paths = {}

        # Save episode-level CSV
        csv_path = self.output_dir / f"{base_name}_episodes.csv"
        df = self.to_dataframe()
        df.to_csv(csv_path, index=False)
        paths["episodes_csv"] = csv_path

        # Save aggregated JSON
        json_path = self.output_dir / f"{base_name}_summary.json"
        summary = asdict(self.get_aggregated_metrics())
        summary["run_id"] = self.run_id
        summary["timestamp"] = datetime.now().isoformat()

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        paths["summary_json"] = json_path

        return paths

    def print_summary(self) -> None:
        """Print a summary of the benchmark results."""
        metrics = self.get_aggregated_metrics()

        print("\n" + "=" * 60)
        print(f"BENCHMARK RESULTS: {metrics.policy}")
        print(f"Task: {metrics.task}")
        print("=" * 60)
        print(f"Total Episodes:      {metrics.total_episodes}")
        print(f"Successful Episodes: {metrics.success_count}")
        print(f"Success Rate:        {metrics.success_rate:.1%}")
        print("-" * 60)

        if metrics.avg_completion_time is not None:
            print(f"Avg Completion Time: {metrics.avg_completion_time:.2f}s "
                  f"(+/- {metrics.std_completion_time:.2f}s)")
        print(f"Avg Steps:           {metrics.avg_steps:.1f}")
        print(f"Avg Path Length:     {metrics.avg_path_length:.3f}m")
        print(f"Grasp Success Rate:  {metrics.grasp_success_rate:.1%}")

        if metrics.avg_final_distance is not None:
            print(f"Avg Final Distance:  {metrics.avg_final_distance:.3f}m")
        print("=" * 60 + "\n")


class ComparisonReport:
    """
    Generate comparison reports across multiple policies.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.results: list[BenchmarkMetrics] = []

    def add_result(self, metrics: BenchmarkMetrics) -> None:
        """Add a benchmark result for comparison."""
        self.results.append(metrics)

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all results."""
        data = []
        for m in self.results:
            data.append({
                "Policy": m.policy,
                "Task": m.task,
                "Success Rate": f"{m.success_rate:.1%}",
                "Avg Time (s)": f"{m.avg_completion_time:.2f}" if m.avg_completion_time else "N/A",
                "Avg Steps": f"{m.avg_steps:.0f}",
                "Grasp Rate": f"{m.grasp_success_rate:.1%}",
            })

        return pd.DataFrame(data)

    def save_comparison(self, filename: str = "comparison") -> Path:
        """Save comparison report."""
        df = self.generate_comparison_table()

        # Save CSV
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)

        # Save JSON with full metrics
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump([asdict(m) for m in self.results], f, indent=2)

        return csv_path

    def print_comparison(self) -> None:
        """Print comparison table to console."""
        df = self.generate_comparison_table()
        print("\n" + "=" * 80)
        print("POLICY COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80 + "\n")


def plot_success_rates(
    results: list[BenchmarkMetrics],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot success rates as a bar chart.

    Args:
        results: List of benchmark results
        output_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    policies = [r.policy for r in results]
    rates = [r.success_rate * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(policies, rates, color='steelblue')

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Policy Comparison: Success Rates')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()
