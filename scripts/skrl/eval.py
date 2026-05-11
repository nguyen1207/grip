# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a skrl checkpoint and report pick-and-place success rate.

Success criterion: object lifted (z > minimal_height) AND within `success_threshold`
of the goal at any point during the episode.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a skrl checkpoint with success-rate metric.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments for evaluation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes to evaluate.")
parser.add_argument("--success_threshold", type=float, default=0.01, help="Distance (m) to count as success.")
parser.add_argument("--minimal_height", type=float, default=0.04, help="Minimum object z to count as lifted.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--agent", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Grip.tasks  # noqa: F401

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Run evaluation."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
        raw_env = env.unwrapped

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    num_envs = args_cli.num_envs
    device = raw_env.device

    success_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)
    lift_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_success_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    last_distance = torch.zeros(num_envs, dtype=torch.float, device=device)

    completed_episodes = 0
    successful_episodes = 0
    lifted_episodes = 0
    sum_first_success_step = 0.0
    n_first_success_recorded = 0
    sum_final_distance = 0.0

    print(f"[INFO] Evaluating {args_cli.num_episodes} episodes with {num_envs} parallel envs.")
    print(f"[INFO] Success: object within {args_cli.success_threshold} m of goal AND lifted (z > {args_cli.minimal_height}).")

    while completed_episodes < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

        object_pos_w = raw_env.scene["object"].data.root_pos_w
        robot = raw_env.scene["robot"]
        command = raw_env.command_manager.get_command("object_pose")
        des_pos_b = command[:, :3]
        des_pos_w, _ = math_utils.combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
        )
        distance = torch.norm(object_pos_w - des_pos_w, dim=1)
        lifted = object_pos_w[:, 2] > args_cli.minimal_height
        at_goal = (distance < args_cli.success_threshold) & lifted

        new_success = at_goal & ~success_flag
        first_success_step = torch.where(new_success, step_count, first_success_step)
        success_flag = success_flag | at_goal
        lift_flag = lift_flag | lifted
        last_distance = distance
        step_count = step_count + 1

        done = (terminated.bool() | truncated.bool()).view(-1)
        n_done = int(done.sum().item())
        if n_done > 0:
            n_success = int((success_flag & done).sum().item())
            n_lift = int((lift_flag & done).sum().item())
            done_succ_mask = done & success_flag & (first_success_step >= 0)
            if done_succ_mask.any():
                sum_first_success_step += float(first_success_step[done_succ_mask].float().sum().item())
                n_first_success_recorded += int(done_succ_mask.sum().item())
            sum_final_distance += float(last_distance[done].sum().item())

            completed_episodes += n_done
            successful_episodes += n_success
            lifted_episodes += n_lift

            success_flag[done] = False
            lift_flag[done] = False
            first_success_step[done] = -1
            step_count[done] = 0

            rate = 100.0 * successful_episodes / completed_episodes
            print(f"[EVAL] episodes={completed_episodes}/{args_cli.num_episodes}  success={successful_episodes}  rate={rate:.1f}%")

    succ_rate = 100.0 * successful_episodes / max(completed_episodes, 1)
    lift_rate = 100.0 * lifted_episodes / max(completed_episodes, 1)
    mean_first_success = sum_first_success_step / max(n_first_success_recorded, 1)
    mean_final_distance = sum_final_distance / max(completed_episodes, 1)

    print("\n========== EVALUATION RESULT ==========")
    print(f"Episodes evaluated:        {completed_episodes}")
    print(f"Success rate (<{args_cli.success_threshold:.3f} m): {succ_rate:.2f}%  ({successful_episodes}/{completed_episodes})")
    print(f"Lift rate (z>{args_cli.minimal_height:.3f}):     {lift_rate:.2f}%  ({lifted_episodes}/{completed_episodes})")
    print(f"Mean steps to first success:  {mean_first_success:.1f}  (over {n_first_success_recorded} successful eps)")
    print(f"Mean final distance to goal:  {mean_final_distance*1000:.1f} mm")
    print("=======================================")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
