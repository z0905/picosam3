import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
import wandb

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from .utils import pad_bbox_to_square

IMG_ROOT = "dataset/train2017"
ANN_FILE = "dataset/annotations/instances_train2017.json"

CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_tiny.pt"
CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t.yaml"

IMAGE_SIZE = 96
CACHE_DIR = "dataset/teacher_sam2_logits"
os.makedirs(CACHE_DIR, exist_ok=True)

MIN_ROI_SIZE = 4
VIS_INTERVAL = 1500  


def cache_path_for_ann(ann_id: int):
    return os.path.join(CACHE_DIR, f"ann_{ann_id}.pt")


def visualize_wandb(image, roi, teacher_mask, ann_id):
    rx1, ry1, rx2, ry2 = roi
    roi_h, roi_w = ry2 - ry1, rx2 - rx1

    
    teacher_mask_t = torch.tensor(teacher_mask).unsqueeze(0).unsqueeze(0)
    teacher_mask_resized = F.interpolate(
        teacher_mask_t,
        size=(roi_h, roi_w),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    overlay = image.copy()
    overlay[ry1:ry2, rx1:rx2, 1] = np.clip(
        overlay[ry1:ry2, rx1:rx2, 1] + teacher_mask_resized * 150, 0, 255
    )

    wandb.log({
        "teacher_visualization": wandb.Image(
            overlay,
            caption=f"ann_id={ann_id}"
        )
    })



def main():
    wandb.init(
        project="PicoSAM2-teacher-cache",
        name="sam2_binary_mask_teacher",
        config={
            "image_size": IMAGE_SIZE,
            "teacher": "sam2.1_hiera_tiny",
            "mask_type": "binary_image_space",
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = build_sam2(
        CONFIG_PATH,
        CHECKPOINT_PATH,
        device=device,
        mode="eval"
    )
    predictor = SAM2ImagePredictor(teacher_model)

    coco = COCO(ANN_FILE)

    anns = [
        ann for ann in coco.loadAnns(coco.getAnnIds())
        if "segmentation" in ann and ann.get("iscrowd", 0) == 0
    ]

    anns_by_img = {}
    for ann in anns:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    ann_counter = 0
    processed_counter = 0
    skipped_cached = 0
    skipped_small = 0
    skipped_empty = 0
    first_visualization_done = False

    print(f"Total annotations to process: {len(anns)}")
    print(f"Total images: {len(anns_by_img)}")
    
    
    wandb.log({
        "total_annotations": len(anns),
        "total_images": len(anns_by_img),
        "status": "started"
    })

    with torch.no_grad():
        for image_id, ann_list in tqdm(
            anns_by_img.items(),
            desc="Caching teacher masks"
        ):
            img_info = coco.loadImgs(image_id)[0]
            img_path = os.path.join(
                IMG_ROOT,
                img_info.get("file_name", f"{image_id:012d}.jpg")
            )

            image = Image.open(img_path).convert("RGB")
            img_np = np.array(image)
            img_h, img_w = img_np.shape[:2]

            predictor.set_image(img_np)

            for ann in ann_list:
                out_path = cache_path_for_ann(ann["id"])
                if os.path.exists(out_path):
                    skipped_cached += 1
                    
                    if skipped_cached % 1000 == 0:
                        print(f"Skipped {skipped_cached} cached annotations so far...")
                        wandb.log({"skipped_cached": skipped_cached})
                    continue

                
                bx, by, bw, bh = ann["bbox"]
                box = np.array([[bx, by, bx + bw, by + bh]], dtype=np.float32)

                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=False
                )

                
                
                teacher_mask_full = masks[0].astype(np.float32)  
                teacher_score = float(scores[0])
                
                
                rx1, ry1, rx2, ry2 = pad_bbox_to_square(
                    ann["bbox"], img_w, img_h, padding=0.1
                )

                if (rx2 - rx1) < MIN_ROI_SIZE or (ry2 - ry1) < MIN_ROI_SIZE:
                    skipped_small += 1
                    continue

                roi_mask = teacher_mask_full[ry1:ry2, rx1:rx2]

                if roi_mask.size == 0 or roi_mask.shape[0] == 0 or roi_mask.shape[1] == 0:
                    skipped_empty += 1
                    continue

                
                
                
                roi_mask_t = torch.tensor(roi_mask).unsqueeze(0).unsqueeze(0)
                roi_mask_t = F.interpolate(
                    roi_mask_t,
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)  
                
                
                eps = 1e-7
                roi_mask_clamped = torch.clamp(roi_mask_t, eps, 1 - eps)
                roi_logits = torch.log(roi_mask_clamped / (1 - roi_mask_clamped))

                payload = {
                    "logits": roi_logits.to(torch.float16).cpu(),  
                    "score": teacher_score,
                    "bbox": ann["bbox"],
                    "roi": (rx1, ry1, rx2, ry2),
                    "image_id": int(image_id),
                }

                torch.save(payload, out_path)

                processed_counter += 1
                ann_counter += 1

                
                if processed_counter == 1:
                    print(f"Creating first visualization (ann_id={ann['id']})")
                    
                    roi_mask_vis = torch.sigmoid(roi_logits.squeeze()).cpu().numpy()
                    visualize_wandb(
                        image=img_np,
                        roi=(rx1, ry1, rx2, ry2),
                        teacher_mask=roi_mask_vis,
                        ann_id=ann["id"]
                    )
                    first_visualization_done = True

                
                should_log = (
                    processed_counter <= 10 or 
                    (processed_counter <= 100 and processed_counter % 10 == 0) or
                    processed_counter % 100 == 0
                )
                
                if should_log:
                    total_checked = ann_counter + skipped_cached + skipped_small + skipped_empty
                    wandb.log({
                        "processed": processed_counter,
                        "skipped_cached": skipped_cached,
                        "skipped_small": skipped_small,
                        "skipped_empty": skipped_empty,
                        "total_checked": total_checked,
                        "progress_pct": (total_checked / len(anns)) * 100,
                    })
                    print(f"Progress: {processed_counter} processed, {total_checked}/{len(anns)} checked ({total_checked/len(anns)*100:.1f}%)")
                
                if processed_counter > 1 and processed_counter % VIS_INTERVAL == 0:
                    print(f"Creating visualization #{processed_counter // VIS_INTERVAL} (ann_id={ann['id']})")
                    
                    roi_mask_vis = torch.sigmoid(roi_logits.squeeze()).cpu().numpy()
                    visualize_wandb(
                        image=img_np,
                        roi=(rx1, ry1, rx2, ry2),
                        teacher_mask=roi_mask_vis,
                        ann_id=ann["id"]
                    )

    
    wandb.log({
        "final_processed": processed_counter,
        "final_skipped_cached": skipped_cached,
        "final_skipped_small": skipped_small,
        "final_skipped_empty": skipped_empty,
        "final_total": processed_counter + skipped_cached + skipped_small + skipped_empty,
    })

    print(f"\n=== Teacher Precompute Summary ===")
    print(f"Processed: {processed_counter}")
    print(f"Skipped (cached): {skipped_cached}")
    print(f"Skipped (too small): {skipped_small}")
    print(f"Skipped (empty ROI): {skipped_empty}")
    print(f"Total: {processed_counter + skipped_cached + skipped_small + skipped_empty}")

    wandb.finish()


if __name__ == "__main__":
    main()
