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
TEXT_PROMPT = "car. tire."
IMG_PATH = "notebooks/images/truck.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True
MULTIMASK_OUTPUT = False

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT
img_path = IMG_PATH

image_source, image = load_image(img_path)

# >>> changed: ensure arrays we pass around are writable/contiguous
if hasattr(image_source, "flags") and not image_source.flags.writeable:
    image_source = image_source.copy()
if hasattr(image, "flags") and not image.flags.writeable:
    image = image.copy()

sam2_predictor.set_image(image_source)

boxes, confidences, class_names = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE
)

# >>> changed: handle no detections early
if boxes is None or len(boxes) == 0:
    print("No detections from GroundingDINO at the given thresholds.")
    # Optionally: save a copy of the input image and exit
    img0 = cv2.imread(img_path)
    if img0 is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "no_detections.jpg"), img0)
    exit(0)

# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
# >>> changed: ensure float32 + contiguous for SAM2
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").to("cpu").to(torch.float32).contiguous().numpy()

# >>> changed: move autocast into a proper context, and scope it to SAM2 only
use_bf16 = (DEVICE == "cuda")
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    # Ampere TF32 (Jetson Orin is sm_87)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

with torch.autocast(device_type=("cuda" if use_bf16 else "cpu"), dtype=torch.bfloat16 if use_bf16 else torch.float32):
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=MULTIMASK_OUTPUT,
    )

"""
Sample the best mask according to the score
"""
if MULTIMASK_OUTPUT and masks is not None and masks.size > 0:
    best = np.argmax(scores, axis=1)
    masks = masks[np.arange(masks.shape[0]), best]

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

confidences = confidences.numpy().tolist()
class_ids = np.arange(len(class_names))

labels = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]

"""
Visualize image with supervision useful API
"""
img = cv2.imread(img_path)
# >>> changed: guard when masks are empty to avoid bool() errors
has_masks = masks is not None and masks.size > 0
detections = sv.Detections(
    xyxy=input_boxes,                      # (n, 4)
    mask=(masks.astype(bool) if has_masks else None),  # (n, h, w) or None
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

if has_masks:
    mask_annotator = sv.MaskAnnotator()
    annotated_frame_mask = mask_annotator.annotate(scene=annotated_frame.copy(), detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame_mask)

"""
Dump the results in standard format and save as json files
"""
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    mask_rles = [single_mask_to_rle(mask) for mask in (masks if has_masks else [])]
    results = {
        "image_path": img_path,
        "annotations": [
            {
                "class_name": cls,
                "bbox": box,
                "segmentation": rle if has_masks else None,
                "score": float(score),
            }
            for cls, box, rle, score in zip(class_names, input_boxes.tolist(), mask_rles, scores.tolist() if hasattr(scores, "tolist") else scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_local_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)
