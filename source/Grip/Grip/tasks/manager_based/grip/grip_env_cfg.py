# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pink.tasks import DampingTask, FrameTask

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FRAME_MARKER_CFG,
    OffsetCfg,
)

from . import mdp

from .ur_gripper import UR_GRIPPER_CFG

##
# Scene definition
##

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"


@configclass
class GripSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # robot
    robot = UR_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur10_instanceable/base_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ur10_instanceable/ee_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.2, 0.0, 0.0],
                ),
            ),
        ],
    )

    # ee_frame = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/ur10_instanceable/base_link",
    #     debug_vis=False,
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/Robot/ur10_instanceable/ee_link",
    #             name="end_effector",
    #         ),
    #     ],
    # )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0, 0.055), rot=(1, 0, 0, 0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2), metallic=0.2
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.5,
        use_default_offset=True,
        debug_vis=True,
    )

    # gripper_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "finger_joint",
    #         "right_outer_knuckle_joint",
    #         # "left_inner_knuckle_joint",
    #         # "right_inner_knuckle_joint",
    #         "left_outer_finger_joint",
    #         "right_outer_finger_joint",
    #         "left_inner_finger_joint",
    #         "right_inner_finger_joint",
    #         "right_inner_finger_pad_joint",
    #         "left_inner_finger_pad_joint",
    #     ],
    #     open_command_expr={
    #         "finger_joint": 0.7,
    #         "right_outer_knuckle_joint": 0.7,
    #         # "left_inner_knuckle_joint": 0.7,
    #         # "right_inner_knuckle_joint": 0.7,
    #         "left_outer_finger_joint": 0.7,
    #         "right_outer_finger_joint": 0.7,
    #         "left_inner_finger_joint": 0.7,
    #         "right_inner_finger_joint": 0.7,
    #         "right_inner_finger_pad_joint": 0.7,
    #         "left_inner_finger_pad_joint": 0.7,
    #     },
    #     close_command_expr={
    #         "finger_joint": 0.0,
    #         "right_outer_knuckle_joint": 0.0,
    #         # "left_inner_knuckle_joint": 0.0,
    #         # "right_inner_knuckle_joint": 0.0,
    #         "left_outer_finger_joint": 0.0,
    #         "right_outer_finger_joint": 0.0,
    #         "left_inner_finger_joint": 0.0,
    #         "right_inner_finger_joint": 0.0,
    #         "right_inner_finger_pad_joint": 0.0,
    #         "left_inner_finger_pad_joint": 0.0,
    #     },
    #     debug_vis=True,
    # )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "finger_joint",
            "right_outer_knuckle_joint",
            "right_outer_finger_joint",
            "right_inner_finger_joint",
            # "right_inner_finger_knuckle_joint",
            "left_outer_finger_joint",
            # "left_inner_finger_knuckle_joint",
            "left_inner_finger_joint",
        ],
        open_command_expr={
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            # "right_inner_finger_knuckle_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            # "left_inner_finger_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
        },
        close_command_expr={
            "finger_joint": 0.785398163,
            "right_outer_knuckle_joint": 0.785398163,
            "right_outer_finger_joint": 0.0,
            "right_inner_finger_joint": 0.785398163,  # 0.785398163,
            # "right_inner_finger_knuckle_joint": -0.785398163,
            "left_outer_finger_joint": 0.0,
            # "left_inner_finger_knuckle_joint": -0.785398163,
            "left_inner_finger_joint": -0.785398163,  # -0.785398163,
        },
    )


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel
        )
        actions = ObsTerm(func=mdp.last_action)

        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


    # reset_robot_joins = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.75, 1.25),
    #         "velocity_range": (0, 0),
    #     },
    # )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.1, 0.1],
                "y": [-0.25, 0.25],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # override_cube_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "static_friction_range": (1.0, 1.0),
    #         "dynamic_friction_range": (0.8, 0.8),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 1,
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.1
    )

    # facing_object = RewTerm(
    #     func=mdp.object_ee_orientation, weight=1.0
    # )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
            ),
        },
    )

    # joint_deviation_wrist = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.002,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 # "wrist_1_joint",
    #                 "wrist_2_joint",
    #                 "wrist_3_joint",
    #             ],
    #         ),
    #     },
    # )

    joint_vel_at_goal = RewTerm(
        func=mdp.joint_vel_at_goal,
        weight=-5e-2,
        params={
            "command_name": "object_pose",
            "goal_radius": 0.05,
            "minimal_height": 0.04,
        },
    )

    joint_deviation_shoulder_pan = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["shoulder_pan_joint"],
            ),
        },
    )

    # joint_torques = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-1e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # fingers_near_object = RewTerm(
    #     func=mdp.fingers_near_object,
    #     params={"std": 0.1},
    #     weight=2.0,
    # )

    # object_goal_distance_penalty = RewTerm(
    #     func=mdp.object_goal_l2_penalty,
    #     params={"command_name": "object_pose"},
    #     weight=-1.0,
    # )


@configclass
class CurriculumCfg:
    action_rate_1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000},
    )
    # action_rate_2 = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "action_rate", "weight": -5e-2, "num_steps": 60000},
    # )
    # action_rate_3 = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 90000},
    # )

    joint_vel_1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000},
    )
    # joint_vel_2 = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "joint_vel", "weight": -5e-2, "num_steps": 60000},
    # )
    # joint_vel_3 = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 90000},
    # )

    # lifting_decay = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "lifting_object", "weight": 5.0, "num_steps": 60000},
    # )
    # tracking_boost = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "object_goal_tracking",
    #         "weight": 32.0,
    #         "num_steps": 60000,
    #     },
    # )
    fine_boost = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "object_goal_tracking_fine_grained",
            "weight": 25.0,
            "num_steps": 30000,
        },
    )

    # shoulder_pan_pen_boost = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "joint_deviation_shoulder_pan",
    #         "weight": -0.02,
    #         "num_steps": 60000,
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


##
# Environment configuration
##


@configclass
class GripEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: GripSceneCfg = GripSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@configclass
class GripEnvCfg_PLAY(GripEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4
        # disable randomization for play
        self.observations.policy.enable_corruption = False
