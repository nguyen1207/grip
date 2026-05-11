# Grip — Reinforcement Learning for Robotic Grasping and Lifting

PPO policy trained in **NVIDIA Isaac Lab** to grasp a cube and lift it to a randomized target pose using a **UR10 arm** and **Robotiq 2F-140 parallel gripper**.

**Screencast — multi-environment overview:**

![demo](assets/demo.gif)

https://github.com/user-attachments/assets/61ed57f7-55b4-494b-830b-6088f28c5daf

> Can't see the video? [Watch on YouTube](https://www.youtube.com/watch?v=-z8kHYnacOA).

## Results

| Metric | Value |
|---|---|
| Grasp rate | **100%** |
| Goal-reach success (within 1 cm) | **93%** |
| Evaluation episodes | 128 |
| Parallel training environments | 4,096 |

Evaluated with deterministic policy (action mean) on randomized cube and goal poses.

## Stack

- **Simulator**: NVIDIA Isaac Lab / Isaac Sim
- **Algorithm**: PPO via skrl
- **Robot**: UR10 + Robotiq 2F-140 gripper
- **Control**: 6-DOF joint position + binary gripper open/close
- **Logging**: Weights & Biases

## Install

Requires [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) installed.

```bash
python -m pip install -e source/Grip
```

Verify the task is registered:

```bash
python scripts/list_envs.py
```

## Train

```bash
python scripts/skrl/train.py --task Template-Grip-v0 --num_envs 4096
```

Logs and checkpoints write to `logs/skrl/grip_ur10/<timestamp>_ppo_torch/`.

## Play (visualize)

```bash
python scripts/skrl/play.py --task Template-Grip-v0 --num_envs 1 \
    --checkpoint logs/skrl/grip_ur10/<run>/checkpoints/best_agent.pt
```

Record a video clip:

```bash
python scripts/skrl/play.py --task Template-Grip-v0 --num_envs 1 \
    --checkpoint <path> --video --video_length 600 --enable_cameras
```

## Evaluate (success rate)

```bash
python scripts/skrl/eval.py --task Template-Grip-v0 --num_envs 64 \
    --checkpoint <path> --num_episodes 128 --success_threshold 0.01
```

Reports success rate, lift rate, time-to-success, and final distance to goal.

## Engineering notes

- **Rewards**: the robot gets points for getting close to the cube (`reaching_object`), lifting it (`lifting_object`), and moving it toward the goal (`object_goal_tracking` and `object_goal_tracking_fine_grained`). It loses points for jerky motion (`action_rate`), fast joint movement (`joint_vel`), and turning the shoulder too far (`joint_deviation_shoulder_pan`). A small extra penalty (`joint_vel_at_goal`) kicks in only when the cube is already lifted and close to the goal, to keep the arm still at the end.
- **Shaking fix**: at first the arm kept shaking once it reached the goal. By looking at the action plots in W&B, I saw the policy kept changing its target by small amounts every step. Adding `joint_vel_at_goal` and raising the joint damping (40 → 120) cut the shaking by about 38% without making the policy worse at the task.
- **Physics**: tuned the cube friction, solver settings, and joint stiffness/damping so the gripper can grip the cube firmly without it slipping or sticking.
