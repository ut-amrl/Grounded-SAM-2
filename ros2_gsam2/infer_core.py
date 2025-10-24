# ros2_gsam2/infer_core.py
import time
from typing import Dict, Any, Tuple

import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision.ops import box_convert

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _single_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _normalize_masks_scores(masks, scores, multimask: bool) -> Tuple[bool, np.ndarray, np.ndarray]:
    has_masks = (masks is not None) and (getattr(masks, "size", 0) > 0)
    scores = np.array(scores) if scores is not None else None

    if has_masks:
        if masks.ndim == 4:
            # choose best mask per detection by score when available
            if scores is not None and scores.ndim == 2 and scores.shape[0] == masks.shape[0]:
                best = np.argmax(scores, axis=1)
                masks = masks[np.arange(masks.shape[0]), best]
                scores = scores.max(axis=1)
            else:
                masks = masks[:, 0]
                if scores is not None and scores.ndim == 2:
                    scores = scores[:, 0]
        elif masks.ndim == 3:
            if scores is not None and scores.ndim == 2 and scores.shape[0] == masks.shape[0]:
                scores = scores.max(axis=1)
        else:
            has_masks = False

        if has_masks:
            masks = masks.astype(bool)

    if scores is not None:
        scores = scores.astype(np.float32)

    return has_masks, masks, scores


def _ensure_tensor(x, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Accept numpy/list/tuple/torch and return a torch.Tensor on `device`.
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        x_np = np.asarray(x)
        t = torch.from_numpy(x_np)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if t.device != device:
        t = t.to(device)
    return t


class GSAM2Core:
    """
    Minimal, ROS-agnostic wrapper that:
      - loads GroundingDINO + SAM2
      - runs detection + segmentation on a file path or an already-loaded image
      - returns FLAT arrays compatible with your service response
    """

    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        gdino_config: str,
        gdino_checkpoint: str,
        device: str = "cuda",
    ):
        torch.set_grad_enabled(False)
        # normalize torch device
        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device_str = "cuda" if use_cuda else "cpu"
        self.device = torch.device(self.device_str)

        # Build SAM2 predictor
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device_str)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Build GroundingDINO
        self.gdino = load_model(
            model_config_path=gdino_config,
            model_checkpoint_path=gdino_checkpoint,
            device=self.device_str,
        )

        # Ampere (e.g., Orin) TF32 for speed
        if use_cuda and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.sam2_model_name = sam2_checkpoint.split("/")[-1]
        self.gdino_model_name = gdino_checkpoint.split("/")[-1]

    def infer_from_path(
        self,
        img_path: str,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        image_source, image = load_image(img_path)

        # ensure arrays are writable
        if hasattr(image_source, "flags") and not image_source.flags.writeable:
            image_source = image_source.copy()
        if hasattr(image, "flags") and not image.flags.writeable:
            image = image.copy()

        return self._infer_impl(
            image_source=image_source,
            gdino_image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            multimask_output=multimask_output,
        )

    # def infer_from_rgb(
    #     self,
    #     rgb_numpy,  # HxWx3, uint8 RGB
    #     text_prompt: str,
    #     box_threshold: float,
    #     text_threshold: float,
    #     multimask_output: bool,
    # ) -> Dict[str, Any]:
    #     image_source = rgb_numpy
    #     gdino_image = rgb_numpy

    #     # ensure writable
    #     if hasattr(image_source, "flags") and not image_source.flags.writeable:
    #         image_source = image_source.copy()
    #     if hasattr(gdino_image, "flags") and not gdino_image.flags.writeable:
    #         gdino_image = gdino_image.copy()

    #     return self._infer_impl(
    #         image_source=image_source,
    #         gdino_image=gdino_image,
    #         text_prompt=text_prompt,
    #         box_threshold=box_threshold,
    #         text_threshold=text_threshold,
    #         multimask_output=multimask_output,
    #     )
    
    def infer_from_rgb(
        self,
        rgb_numpy: np.ndarray,  # HxWx3, uint8 RGB
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """
        Debug/sanity version: dump the RGB array to /tmp/gsam2_input.png
        and reuse the infer_from_path() pipeline.
        """
        # Basic sanity checks
        if not isinstance(rgb_numpy, np.ndarray) or rgb_numpy.ndim != 3 or rgb_numpy.shape[2] != 3:
            raise ValueError(f"Expected RGB numpy HxWx3, got shape={getattr(rgb_numpy, 'shape', None)}")

        # Ensure uint8 RGB
        if rgb_numpy.dtype != np.uint8:
            rgb_numpy = np.clip(rgb_numpy, 0, 255).astype(np.uint8, copy=False)

        # Save to disk (RGB)
        dump_path = "/tmp/gsam2_input.png"
        try:
            from PIL import Image
            Image.fromarray(rgb_numpy, mode="RGB").save(dump_path)
        except Exception as e:
            # Fall back to OpenCV if PIL is unavailable
            try:
                import cv2
                bgr = rgb_numpy[:, :, ::-1]  # RGB -> BGR
                cv2.imwrite(dump_path, bgr)
            except Exception as e2:
                raise RuntimeError(f"Failed to write debug image: PIL error={e}; OpenCV error={e2}")

        # Reuse the file-based path flow
        return self.infer_from_path(
            img_path=dump_path,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            multimask_output=multimask_output,
        )


    def _infer_impl(
        self,
        image_source: np.ndarray,
        gdino_image: np.ndarray,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        h, w, _ = image_source.shape
        self.sam2_predictor.set_image(image_source)

        t0 = time.time()
        # GroundingDINO
        boxes, confidences, class_names = predict(
            model=self.gdino,
            image=gdino_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device_str,
        )

        if boxes is None or len(boxes) == 0:
            return dict(
                n=0,
                img_width=w,
                img_height=h,
                box_format="xyxy",
                rle_encoding="coco_rle",
                x_min=[], y_min=[], x_max=[], y_max=[],
                class_name=[], score=[],
                rle_counts=[], rle_height=[], rle_width=[],
                sam2_model_name=self.sam2_model_name,
                grounding_model_name=self.gdino_model_name,
                inference_time_ms=(time.time() - t0) * 1e3,
            )

        # ---- SAFE BOX PIPELINE (works whether `boxes` is numpy or torch) ----
        # 1) force tensor on the right device/dtype
        boxes_t = _ensure_tensor(boxes, device=self.device, dtype=torch.float32)  # (N,4) cxcywh in [0..1]
        # 2) scale to pixels
        scale = torch.tensor([w, h, w, h], dtype=boxes_t.dtype, device=boxes_t.device)
        boxes_t = boxes_t * scale
        # 3) convert to xyxy (still tensor), then to CPU float32 numpy for SAM2
        xyxy = box_convert(boxes=boxes_t, in_fmt="cxcywh", out_fmt="xyxy")
        xyxy_np = xyxy.to("cpu").contiguous().numpy().astype("float32")

        # SAM2 under autocast (bf16 on CUDA; float32 on CPU)
        use_bf16 = (self.device_str == "cuda")
        with torch.autocast(device_type=("cuda" if use_bf16 else "cpu"),
                            dtype=(torch.bfloat16 if use_bf16 else torch.float32)):
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                # box=xyxy_np,                    # SAM2 expects numpy float32
                box=xyxy,
                multimask_output=multimask_output,
            )

        has_masks, masks, scores = _normalize_masks_scores(masks, scores, multimask_output)

        # confidences → list[float]
        if hasattr(confidences, "detach"):
            confidences = confidences.detach().cpu().numpy()
        if hasattr(confidences, "tolist"):
            confidences = confidences.tolist()

        # Flatten outputs
        n = xyxy_np.shape[0]
        x_min = xyxy_np[:, 0].tolist()
        y_min = xyxy_np[:, 1].tolist()
        x_max = xyxy_np[:, 2].tolist()
        y_max = xyxy_np[:, 3].tolist()

        class_name = list(class_names)
        # Prefer SAM2 score when present; fall back to GDINO confidences if scores is None/empty
        if scores is not None and len(scores) == n:
            score = scores.tolist()
        else:
            score = confidences if isinstance(confidences, list) else [None] * n

        rle_counts, rle_h, rle_w = [], [], []
        if has_masks:
            for i in range(n):
                rle = _single_mask_to_rle(masks[i])
                rle_counts.append(rle["counts"])
                rle_h.append(int(rle["size"][0]))
                rle_w.append(int(rle["size"][1]))
        else:
            rle_counts = [""] * n
            rle_h = [h] * n
            rle_w = [w] * n

        return dict(
            n=n,
            img_width=w,
            img_height=h,
            box_format="xyxy",
            rle_encoding="coco_rle",
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
            class_name=class_name, score=score,
            rle_counts=rle_counts, rle_height=rle_h, rle_width=rle_w,
            sam2_model_name=self.sam2_model_name,
            grounding_model_name=self.gdino_model_name,
            inference_time_ms=(time.time() - t0) * 1e3,
        )
