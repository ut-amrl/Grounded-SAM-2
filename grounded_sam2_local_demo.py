import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict


"""
Hyper parameters
"""
TEXT_PROMPT = "dr pepper soda can."
IMG_PATH = "/home/ros/cobot_ws/robotpoint_input.png"

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True
MULTIMASK_OUTPUT = False  # if True, we pick the best mask/score per detection


# --------------------------- helpers to *fix* warnings ---------------------------

def force_inference_mode():
    """
    Ensure we never build grad graphs (prevents 'requires_grad=True' warning
    and accidental autograd cost in inference).
    """
    torch.set_grad_enabled(False)


def maybe_disable_gradient_checkpointing(model):
    """
    Some libs (HF/others) trigger the 'use_reentrant' checkpoint warning internally.
    When possible, disable gradient checkpointing at the module level.
    This doesn’t suppress — it *changes behavior* to avoid the path that warns.
    """
    # Common HF-style API
    if hasattr(model, "gradient_checkpointing_enable") or hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    # Walk submodules for similar toggles
    for m in model.modules() if hasattr(model, "modules") else []:
        if hasattr(m, "gradient_checkpointing") and isinstance(getattr(m, "gradient_checkpointing"), bool):
            try:
                m.gradient_checkpointing = False
            except Exception:
                pass
        if hasattr(m, "gradient_checkpointing_disable"):
            try:
                m.gradient_checkpointing_disable()
            except Exception:
                pass


def normalize_masks_and_scores(masks, scores, multimask: bool):
    """
    Make masks -> (N, H, W) and scores -> (N,), selecting best per detection when multiple masks.
    """
    has_masks = (masks is not None) and (getattr(masks, "size", 0) > 0)

    if has_masks:
        # masks may be (N, M, H, W) or (N, 1, H, W) or (N, H, W)
        if masks.ndim == 4:
            # leave selection to the score logic below
            pass
        elif masks.ndim == 3:
            pass
        elif masks.ndim == 1:  # rare edge case
            masks = np.expand_dims(masks[0], 0)
        else:
            has_masks = False

    scores = np.array(scores)  # to numpy

    if has_masks:
        if masks.ndim == 4:
            # scores likely (N, M). Choose best mask per detection.
            if scores.ndim == 2 and scores.shape[0] == masks.shape[0]:
                best_idx = np.argmax(scores, axis=1)  # (N,)
                masks = masks[np.arange(masks.shape[0]), best_idx]  # (N, H, W)
                scores = scores.max(axis=1)  # (N,)
            else:
                # shape mismatch; take first mask per detection conservatively
                masks = masks[:, 0]
                if scores.ndim > 1:
                    scores = scores[:, 0]
        elif masks.ndim == 3:
            if scores.ndim == 2 and scores.shape[0] == masks.shape[0]:
                scores = scores.max(axis=1)
            elif scores.ndim == 0:
                scores = np.array([float(scores)])
            elif scores.ndim == 1:
                pass
            else:
                scores = np.squeeze(scores)
                if scores.ndim != 1:
                    scores = np.max(scores, axis=-1)
    else:
        # No masks — make scores sane but optional
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        elif scores.ndim > 1:
            scores = np.max(scores, axis=1)

    if has_masks:
        masks = masks.astype(bool)  # (N, H, W)
    scores = scores.astype(np.float32) if scores is not None else None

    return has_masks, masks, scores


def single_mask_to_rle(mask: np.ndarray):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ----------------------------------- main -----------------------------------

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# hard fixes we can apply from user code
force_inference_mode()

# Build SAM2 predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
# (SAM2 doesn’t typically use HF checkpointing, but safe to try)
maybe_disable_gradient_checkpointing(sam2_model)

# Build GroundingDINO model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)
# try to disable checkpointing on any HF-backed encoders under GDINO
maybe_disable_gradient_checkpointing(grounding_model)

# Input setup
text = TEXT_PROMPT  # lowercased with trailing dots as required by GDINO text matching
img_path = IMG_PATH

image_source, image = load_image(img_path)

# ensure arrays are writable for downstream ops
if hasattr(image_source, "flags") and not image_source.flags.writeable:
    image_source = image_source.copy()
if hasattr(image, "flags") and not image.flags.writeable:
    image = image.copy()

sam2_predictor.set_image(image_source)

# GroundingDINO detections (pure inference)
with torch.inference_mode():
    boxes, confidences, class_names = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

# handle no detections early
if boxes is None or len(boxes) == 0:
    print("No detections from GroundingDINO at the given thresholds.")
    img0 = cv2.imread(img_path)
    if img0 is not None:
        cv2.imwrite(str(OUTPUT_DIR / "no_detections.jpg"), img0)
    raise SystemExit(0)

# process box prompt for SAM2
h, w, _ = image_source.shape
boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").to("cpu").to(torch.float32).contiguous().numpy()

# optional TF32 enable on Ampere (Orin etc.)
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# SAM2 inference under autocast + inference_mode (prevents autograd warnings)
use_bf16 = (DEVICE == "cuda")
with torch.inference_mode():
    with torch.autocast(device_type=("cuda" if use_bf16 else "cpu"),
                        dtype=(torch.bfloat16 if use_bf16 else torch.float32)):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,            # (N, 4) xyxy
            multimask_output=MULTIMASK_OUTPUT,
        )

# Normalize masks/scores into stable shapes and semantics
has_masks, masks, scores = normalize_masks_and_scores(masks, scores, MULTIMASK_OUTPUT)

# confidences from GroundingDINO for labels
confidences = confidences.numpy().tolist() if hasattr(confidences, "numpy") else list(confidences)
class_ids = np.arange(len(class_names))
labels = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]

# Visualization with supervision
img = cv2.imread(img_path)

detections = sv.Detections(
    xyxy=input_boxes,                        # (N, 4)
    mask=(masks if has_masks else None),     # (N, H, W) or None
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(str(OUTPUT_DIR / "groundingdino_annotated_image.jpg"), annotated_frame)

if has_masks:
    mask_annotator = sv.MaskAnnotator()
    annotated_frame_mask = mask_annotator.annotate(scene=annotated_frame.copy(), detections=detections)
    cv2.imwrite(str(OUTPUT_DIR / "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame_mask)

# JSON dump
if DUMP_JSON_RESULTS:
    N = input_boxes.shape[0]
    anns = []
    for i in range(N):
        seg = single_mask_to_rle(masks[i]) if has_masks else None
        score_i = float(scores[i]) if (scores is not None and i < len(scores)) else None
        anns.append({
            "class_name": class_names[i],
            "bbox": input_boxes[i].tolist(),
            "segmentation": seg,
            "score": score_i,  # SAM2 per-box score
        })

    results = {
        "image_path": img_path,
        "annotations": anns,
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    with open(str(OUTPUT_DIR / "grounded_sam2_local_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)
