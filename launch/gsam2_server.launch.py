# launch/gsam2_server.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# NEW: resolve package share at import time
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

PKG = "ros2_gsam2"
PKG_SHARE = Path(get_package_share_directory(PKG))
RES = PKG_SHARE / "resources"

SAM2_CONFIG_DEF      = str(RES / "configs/sam2.1/sam2.1_hiera_l.yaml")
SAM2_CKPT_DEF        = str(RES / "checkpoints/sam2.1_hiera_large.pt")
GDINO_CONFIG_DEF     = str(RES / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GDINO_CKPT_DEF       = str(RES / "gdino_checkpoints/groundingdino_swint_ogc.pth")

def generate_launch_description():
    # ---- Launch args (with share-based defaults) ----
    sam2_config = DeclareLaunchArgument("sam2_config", default_value=SAM2_CONFIG_DEF)
    sam2_checkpoint = DeclareLaunchArgument("sam2_checkpoint", default_value=SAM2_CKPT_DEF)
    gdino_config = DeclareLaunchArgument("gdino_config", default_value=GDINO_CONFIG_DEF)
    gdino_checkpoint = DeclareLaunchArgument("gdino_checkpoint", default_value=GDINO_CKPT_DEF)

    device = DeclareLaunchArgument("device", default_value="cuda")
    box_threshold = DeclareLaunchArgument("box_threshold", default_value="0.35")
    text_threshold = DeclareLaunchArgument("text_threshold", default_value="0.45")
    multimask_output = DeclareLaunchArgument("multimask_output", default_value="false")
    node_name = DeclareLaunchArgument("node_name", default_value="gsam2_infer_server")
    namespace = DeclareLaunchArgument("namespace", default_value="")

    # optional debug viz (kept as before)
    viz_save_path = DeclareLaunchArgument("viz_save_path", default_value="/tmp/gsam2_debug.png")
    viz_publish_topic = DeclareLaunchArgument("viz_publish_topic", default_value="/gsam2/debug_image")

    node = Node(
        package=PKG,
        executable="infer_server",
        name=LaunchConfiguration("node_name"),
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[{
            "sam2_config": LaunchConfiguration("sam2_config"),
            "sam2_checkpoint": LaunchConfiguration("sam2_checkpoint"),
            "gdino_config": LaunchConfiguration("gdino_config"),
            "gdino_checkpoint": LaunchConfiguration("gdino_checkpoint"),
            "device": LaunchConfiguration("device"),
            "box_threshold": LaunchConfiguration("box_threshold"),
            "text_threshold": LaunchConfiguration("text_threshold"),
            "multimask_output": LaunchConfiguration("multimask_output"),
            "viz_save_path": LaunchConfiguration("viz_save_path"),
            "viz_publish_topic": LaunchConfiguration("viz_publish_topic"),
        }],
    )

    return LaunchDescription([
        sam2_config, sam2_checkpoint, gdino_config, gdino_checkpoint,
        device, box_threshold, text_threshold, multimask_output,
        node_name, namespace, viz_save_path, viz_publish_topic, node,
    ])
