# SO100 Simulation Environment

Simulation environment for testing and benchmarking LeRobot policies with the SO100 robot arm.

## Overview

This project uses [gym-lowcostrobot](https://github.com/jpata/gym-so100) (MuJoCo-based) to simulate the SO100 robot arm and evaluate trained policies.

## Features

- **Automated Testing**: Run hundreds of episodes automatically
- **Success Metrics**: Track success rate, completion time, path efficiency
- **Task Randomization**: Randomize object positions for robust evaluation
- **LeRobot Integration**: Direct integration with trained lerobot policies
- **Batch Benchmarking**: Compare multiple policies side-by-side

## Installation

```bash
# From the lerobot-scratch directory
cd simulation
pip install -r requirements.txt
```

## Available Tasks

| Task | Description |
|------|-------------|
| `PickPlaceCube-v0` | Pick up cube and place at target location |
| `LiftCube-v0` | Lift cube to target height |
| `PushCube-v0` | Push cube to target position |
| `ReachCube-v0` | Move end-effector to target position |
| `StackTwoCubes-v0` | Stack two cubes |

## Usage

### Quick Test
```bash
python run_evaluation.py --policy danbhf/smolVla_so100_pick_place --task PickPlaceCube-v0 --episodes 10
```

### Full Benchmark
```bash
python benchmark_runner.py --config configs/benchmark_config.yaml
```

### Interactive Visualization
```bash
python visualize.py --policy danbhf/act_so100_pick_place --task PickPlaceCube-v0
```

## Project Structure

```
simulation/
├── configs/
│   └── benchmark_config.yaml    # Benchmark configuration
├── src/
│   ├── env_wrapper.py           # Gymnasium environment wrapper
│   ├── policy_interface.py      # LeRobot policy loader
│   ├── task_evaluator.py        # Success detection
│   ├── metrics.py               # Performance metrics
│   └── utils.py                 # Utilities
├── scripts/
│   ├── run_evaluation.py        # Single policy evaluation
│   ├── benchmark_runner.py      # Batch benchmarking
│   └── visualize.py             # Interactive visualization
├── results/                     # Benchmark results
├── requirements.txt
└── README.md
```

## Metrics Collected

- **Success Rate**: Percentage of episodes where task was completed
- **Completion Time**: Time to complete successful episodes
- **Path Efficiency**: Ratio of optimal to actual path length
- **Grasp Success**: For pick tasks, whether object was grasped
- **Placement Accuracy**: Distance from target placement position

## Configuration

See `configs/benchmark_config.yaml` for all options:

```yaml
tasks:
  - name: PickPlaceCube-v0
    episodes: 100
    max_steps: 500
    randomize_positions: true

policies:
  - name: danbhf/act_so100_pick_place
    type: act
  - name: danbhf/smolVla_so100_pick_place
    type: smolvla

success_criteria:
  cube_at_target_threshold: 0.05  # 5cm tolerance
```
