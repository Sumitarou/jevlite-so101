"""SO-101 on the work table in Gazebo Harmonic, server only (headless). No GUI is started.

    source /opt/ros/jazzy/setup.bash && source sim_ws/install/setup.bash && \
      timeout 120 ros2 launch so101_gazebo sim.launch.py [physics_engine:=bullet-featherstone]

Settings come from the repository's configs/sim.yaml and configs/sim_controllers.yaml. The
repository root is found through this file (colcon build --symlink-install) or JEVLITE_ROOT.
`physics_engine` overrides configs/sim.yaml physics.engine (empty: use the config).
If the SO-101 meshes have not been fetched, the launch stops before starting Gazebo.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

FETCH_HINT = (
    "SO-101 meshes are missing: run `python3 scripts/sim/fetch_so101_assets.py` in the "
    "repository, then `colcon build --symlink-install` in sim_ws, and launch again. "
    "（メッシュがありません。scripts/sim/fetch_so101_assets.py を実行してください）"
)


def repo_root() -> Path:
    env = os.environ.get("JEVLITE_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()  # symlink-install: points into sim_ws/src/so101_gazebo/launch
    for parent in here.parents:
        if (parent / "configs" / "sim.yaml").exists():
            return parent
    raise RuntimeError("cannot find the repository (configs/sim.yaml); set JEVLITE_ROOT")


def missing_meshes(desc_share: Path) -> list[str]:
    lock = json.loads((desc_share / "assets.lock").read_text())
    return [n for n in lock["meshes"] if not (desc_share / "meshes" / n).is_file()]


def setup(context, *args, **kwargs):
    root = repo_root()
    desc_share = Path(get_package_share_directory("so101_description"))
    missing = missing_meshes(desc_share)
    if missing:
        raise RuntimeError(f"{FETCH_HINT} Missing: {', '.join(missing[:3])} ...")

    sim_cfg = root / "configs" / "sim.yaml"
    ctrl_cfg = root / "configs" / "sim_controllers.yaml"
    share = Path(get_package_share_directory("so101_gazebo"))
    gen = root / "outputs" / "sim" / "generated"  # git-ignored; kept for inspection
    gen.mkdir(parents=True, exist_ok=True)

    engine = LaunchConfiguration("physics_engine").perform(context)
    world = xacro.process_file(
        str(share / "worlds" / "table.sdf.xacro"),
        mappings={"sim_config": str(sim_cfg), "physics_engine": engine},
    ).toprettyxml(indent="  ")
    world_file = gen / "table.sdf"
    world_file.write_text(world)
    robot = xacro.process_file(
        str(share / "urdf" / "so101_gazebo.urdf.xacro"),
        mappings={"sim_config": str(sim_cfg), "controllers_file": str(ctrl_cfg)},
    ).toxml()
    cfg = yaml.safe_load(sim_cfg.read_text())
    sys.path.insert(0, str(root))  # scripts.sim.urdf_tools (standard library only)
    from scripts.sim.urdf_tools import recolor

    robot = recolor(robot, cfg["appearance"])  # white arm, grey base (configs/sim.yaml)
    (gen / "so101_gazebo.urdf").write_text(robot)

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("ros_gz_sim")) / "launch" / "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": f"-s -r --headless-rendering -v 2 {world_file}",
            "on_exit_shutdown": "true",
        }.items(),
    )
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot, "use_sim_time": True}],
        output="screen",
    )
    create = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "robot_description", "-name", "so101", "-allow_renaming", "false"],
        output="screen",
    )
    camera_topic = cfg["camera"]["topic"]
    block = cfg["block"]
    gz_block_pose = f"/model/{block['name']}/pose"  # PosePublisher of the block model
    intruder = cfg["intruder"]
    gz_intruder_pose = f"/model/{intruder['name']}/pose"  # M4a: the rod's PosePublisher
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            f"{camera_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            f"{gz_block_pose}@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
            f"{gz_intruder_pose}@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
            "/world/table/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
        remappings=[
            (gz_block_pose, block["pose_topic"]),
            (gz_intruder_pose, intruder["pose_topic"]),
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                name,
                "--controller-manager",
                "/controller_manager",
                "--controller-manager-timeout",
                "60",
            ],  # fmt: skip
            parameters=[{"use_sim_time": True}],
            output="screen",
        )
        for name in (
            "joint_state_broadcaster",
            "arm_position_controller",
            "gripper_position_controller",
        )  # fmt: skip
    ]
    return [
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", str(desc_share.parent)),
        gz,
        rsp,
        bridge,
        create,
        RegisterEventHandler(OnProcessExit(target_action=create, on_exit=spawners)),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("physics_engine", default_value="",
                              description="dartsim / bullet-featherstone (empty: config)"),
        OpaqueFunction(function=setup),
    ])  # fmt: skip
