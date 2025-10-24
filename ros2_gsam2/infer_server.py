# ros2_gsam2/infer_server.py
from __future__ import annotations

import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header

from amrl_msgs.srv import GroundedSAM2Srv

import numpy as np
import cv2

try:
    import pycocotools.mask as mask_util  # only used for optional viz
except Exception:
    mask_util = None

from ament_index_python.packages import get_package_share_directory

from .infer_core import GSAM2Core


def _pkg_share_resources(pkg: str = "ros2_gsam2") -> Path:
    """<install>/share/ros2_gsam2/resources"""
    return Path(get_package_share_directory(pkg)) / "resources"


def _resolve_param_path(p: str, *, rel_hint: str | None = None) -> str:
    """
    Return an absolute existing path for model/config/checkpoint parameters.
    Tries:
      1) as given (absolute/relative)
      2) <share>/resources/<p>
      3) <share>/resources/<rel_hint>/<basename(p)>
    """
    pp = Path(p)
    if pp.is_file():
        return str(pp.resolve())

    share_res = _pkg_share_resources()
    cand = share_res / p
    if cand.is_file():
        return str(cand.resolve())

    if rel_hint:
        cand2 = share_res / rel_hint / pp.name
        if cand2.is_file():
            return str(cand2.resolve())

    # last resort: return as-is (core will throw with a clear error)
    return str(pp)


class GSAM2Server(Node):
    def __init__(self):
        super().__init__("gsam2_infer_server")

        # ---------------- Params ----------------
        self.declare_parameter("sam2_config", "")
        self.declare_parameter("sam2_checkpoint", "")
        self.declare_parameter("gdino_config", "")
        self.declare_parameter("gdino_checkpoint", "")
        self.declare_parameter("device", "cuda")

        # thresholds
        self.declare_parameter("box_threshold", 0.35)
        self.declare_parameter("text_threshold", 0.45)
        self.declare_parameter("multimask_output", False)

        # optional debug viz (set empty to disable)
        self.declare_parameter("viz_save_path", "")
        self.declare_parameter("viz_publish_topic", "")

        # ---- read + resolve paths ----
        sam2_cfg_raw = self.get_parameter("sam2_config").get_parameter_value().string_value
        sam2_ckpt_raw = self.get_parameter("sam2_checkpoint").get_parameter_value().string_value
        gdino_cfg_raw = self.get_parameter("gdino_config").get_parameter_value().string_value
        gdino_ckpt_raw = self.get_parameter("gdino_checkpoint").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value

        if not (sam2_cfg_raw and sam2_ckpt_raw and gdino_cfg_raw and gdino_ckpt_raw):
            raise RuntimeError("sam2_config/sam2_checkpoint/gdino_config/gdino_checkpoint are required.")

        sam2_cfg = _resolve_param_path(sam2_cfg_raw, rel_hint="configs")
        sam2_ckpt = _resolve_param_path(sam2_ckpt_raw, rel_hint="checkpoints")
        gdino_cfg = _resolve_param_path(gdino_cfg_raw, rel_hint="grounding_dino/groundingdino/config")
        gdino_ckpt = _resolve_param_path(gdino_ckpt_raw, rel_hint="gdino_checkpoints")

        for k, v in [
            ("sam2_config", sam2_cfg), ("sam2_checkpoint", sam2_ckpt),
            ("gdino_config", gdino_cfg), ("gdino_checkpoint", gdino_ckpt)
        ]:
            if not Path(v).is_file():
                self.get_logger().error(f"[path] {k} not found: {v}")
            else:
                self.get_logger().info(f"[path] {k} = {v}")

        # ---------------- Core ----------------
        self.core = GSAM2Core(
            sam2_config=sam2_cfg,
            sam2_checkpoint=sam2_ckpt,
            gdino_config=gdino_cfg,
            gdino_checkpoint=gdino_ckpt,
            device=device,
        )
        self.bridge = CvBridge()

        # ---------------- Optional viz ----------------
        self.viz_save_path = self.get_parameter("viz_save_path").get_parameter_value().string_value or ""
        viz_topic = self.get_parameter("viz_publish_topic").get_parameter_value().string_value or ""
        self.viz_pub = None
        if viz_topic:
            self.viz_pub = self.create_publisher(ImageMsg, viz_topic, 1)
            self.get_logger().info(f"[viz] Publishing debug images to: {viz_topic}")
        if self.viz_save_path:
            self.get_logger().info(f"[viz] Saving debug image to: {self.viz_save_path}")

        # ---------------- Service ----------------
        self.srv = self.create_service(GroundedSAM2Srv, "gsam2/infer", self.handle_req)
        self.get_logger().info("GSAM2 service ready: /gsam2/infer")

    # ---------------- Service handler ----------------
    def handle_req(self, req: GroundedSAM2Srv.Request, resp: GroundedSAM2Srv.Response):
        # thresholds from request (fallback to node params if <= 0)
        box_thr = req.box_threshold if req.box_threshold > 0.0 else \
            self.get_parameter("box_threshold").get_parameter_value().double_value
        text_thr = req.text_threshold if req.text_threshold > 0.0 else \
            self.get_parameter("text_threshold").get_parameter_value().double_value
        # note: bool is not tri-state in ROS; this keeps your original behavior
        multimask = bool(req.multimask_output) if hasattr(req, "multimask_output") else \
            self.get_parameter("multimask_output").get_parameter_value().bool_value

        # ---- Image: ROS -> RGB numpy ----
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(req.image, desired_encoding="bgr8")
            rgb = cv_bgr[:, :, ::-1].copy()  # BGR -> RGB
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            self._fill_empty(resp, req.image.header if hasattr(req.image, "header") else Header())
            return resp

        # ---- Inference ----
        try:
            t0 = time.time()
            out = self.core.infer_from_rgb(
                rgb_numpy=rgb,
                text_prompt=req.text_prompt,
                box_threshold=float(box_thr),
                text_threshold=float(text_thr),
                multimask_output=bool(multimask),
            )
            # if core didn't set timing, add it
            out.setdefault("inference_time_ms", (time.time() - t0) * 1000.0)
        except Exception as e:
            self.get_logger().error(f"inference error: {e}")
            self._fill_empty(resp, req.image.header if hasattr(req.image, "header") else Header())
            return resp

        # ---- Fill Response (flat) ----
        hdr = req.image.header if hasattr(req.image, "header") else Header()
        resp.header = hdr
        resp.img_width = int(out["img_width"])
        resp.img_height = int(out["img_height"])
        resp.box_format = out.get("box_format", "xyxy")
        resp.rle_encoding = out.get("rle_encoding", "coco_rle")
        resp.n = int(out.get("n", 0))

        resp.x_min = [float(v) for v in out.get("x_min", [])]
        resp.y_min = [float(v) for v in out.get("y_min", [])]
        resp.x_max = [float(v) for v in out.get("x_max", [])]
        resp.y_max = [float(v) for v in out.get("y_max", [])]

        resp.class_name = list(out.get("class_name", []))
        resp.score = [float(v) if v is not None else 0.0 for v in out.get("score", [])]

        resp.rle_counts = list(out.get("rle_counts", []))
        resp.rle_height = [int(v) for v in out.get("rle_height", [])]
        resp.rle_width = [int(v) for v in out.get("rle_width", [])]

        # ---- Optional viz (only if requested) ----
        try:
            if self.viz_save_path or self.viz_pub is not None:
                dbg = self._render_debug_image(cv_bgr, resp)
                if self.viz_save_path:
                    cv2.imwrite(self.viz_save_path, dbg)
                if self.viz_pub is not None:
                    msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
                    msg.header = hdr
                    self.viz_pub.publish(msg)
        except Exception as ve:
            self.get_logger().warn(f"[viz] error: {ve}")

        return resp

    # ---------------- Helpers ----------------
    def _fill_empty(self, resp: GroundedSAM2Srv.Response, header: Header):
        resp.header = header
        resp.img_width = 0
        resp.img_height = 0
        resp.box_format = "xyxy"
        resp.rle_encoding = "coco_rle"
        resp.n = 0
        resp.x_min = []
        resp.y_min = []
        resp.x_max = []
        resp.y_max = []
        resp.class_name = []
        resp.score = []
        resp.rle_counts = []
        resp.rle_height = []
        resp.rle_width = []

    def _render_debug_image(self, bgr: np.ndarray, resp: GroundedSAM2Srv.Response) -> np.ndarray:
        """
        Minimal debug overlay:
          - draw boxes + labels
          - if pycocotools available and RLEs present, overlay masks
        """
        img = bgr.copy()
        h, w = img.shape[:2]
        n = int(resp.n)

        # masks (optional)
        if mask_util is not None and n > 0 and len(resp.rle_counts) == n:
            try:
                for counts, mh, mw in zip(resp.rle_counts, resp.rle_height, resp.rle_width):
                    if not counts:
                        continue
                    rle = {"counts": counts.encode("utf-8"), "size": [int(mh), int(mw)]}
                    m = mask_util.decode(rle)  # (H, W) {0,1}
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    # tint
                    img[m.astype(bool)] = (0.7 * img[m.astype(bool)] + 0.3 * np.array([0, 255, 255])).astype(np.uint8)
            except Exception:
                pass

        # boxes + labels
        for i in range(n):
            x1 = int(round(resp.x_min[i]))
            y1 = int(round(resp.y_min[i]))
            x2 = int(round(resp.x_max[i]))
            y2 = int(round(resp.y_max[i]))
            cls = resp.class_name[i] if i < len(resp.class_name) else ""
            sc = resp.score[i] if i < len(resp.score) else 0.0
            label = f"{cls} {sc:.2f}" if cls else f"{sc:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 255), 2)
            # simple filled label box
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytxt = max(0, y1 - 8)
            cv2.rectangle(img, (x1, ytxt - th - 6), (x1 + tw + 6, ytxt), (0, 180, 255), thickness=-1)
            cv2.putText(img, label, (x1 + 3, ytxt - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

        return img


def main():
    rclpy.init()
    node = GSAM2Server()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
