import os 
import numpy as np
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from model_compression.utils import pad_bbox_to_square
from pycocotools.coco import COCO

class PicoSAMDataset(Dataset):
    def __init__(self, image_root, annotation_file, image_size, cache_dir, require_cache=True):
        self.coco = COCO(annotation_file)
        self.image_dir = image_root
        self.image_size = image_size
        self.image_ids = self.coco.getImgIds()
        self.cache_dir = cache_dir
        self.require_cache = require_cache


        # Build a set of existing image files for fast lookup
        existing_images = set()
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info.get("file_name", f"{img_info['id']:012d}.jpg")
            img_path = os.path.join(self.image_dir, file_name)
            if os.path.exists(img_path):
                existing_images.add(img_id)

        all_annotations = [
            ann for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.image_ids))
            if "segmentation" in ann and ann.get("iscrowd", 0) == 0 and ann["image_id"] in existing_images
        ]

        skipped_missing_image = len(self.image_ids) - len(existing_images)
        if skipped_missing_image > 0:
            print(f"Dataset: {skipped_missing_image} images not found in {self.image_dir}")

        if require_cache:
            self.annotations = []
            skipped_no_cache = 0
            for ann in all_annotations:
                cache_path = os.path.join(self.cache_dir, f"ann_{ann['id']}.pt")
                if os.path.exists(cache_path):
                    self.annotations.append(ann)
                else:
                    skipped_no_cache += 1

            print(f"Dataset: {len(self.annotations)} annotations with cached teacher logits")
            print(f"Dataset: {skipped_no_cache} annotations skipped (no cache file)")
        else:
            self.annotations = all_annotations
            print(f"Dataset: {len(self.annotations)} annotations (cache not required)")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]

        
        img_info = self.coco.loadImgs(ann["image_id"])[0]
        file_name = img_info.get("file_name", f"{img_info['id']:012d}.jpg")
        img_path = os.path.join(self.image_dir, file_name)

        
        image = Image.open(img_path).convert("RGB")
        mask = self.coco.annToMask(ann)
        
        img_width, img_height = image.size
        
        
        bbox = ann["bbox"]  
        x1, y1, x2, y2 = pad_bbox_to_square(bbox, img_width, img_height, padding=0.1)
        
        
        image_cropped = image.crop((x1, y1, x2, y2))
        mask_cropped = Image.fromarray(mask).crop((x1, y1, x2, y2))
        
        
        image_rs = image_cropped.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_rs = mask_cropped.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_np = np.array(mask_rs)
        
        
        x = self.image_size // 2
        y = self.image_size // 2

        
        image_tensor = self.transform(image_rs)  
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)  
        prompt_coords = (x, y)

        
        if self.require_cache:
            cache_path = os.path.join(self.cache_dir, f"ann_{ann['id']}.pt")
            cached = torch.load(cache_path, map_location="cpu")

            teacher_logits = cached["logits"]

            assert teacher_logits.shape[-2:] == (self.image_size, self.image_size), \
                f"Teacher logits wrong size: {teacher_logits.shape}"

            teacher_logits = teacher_logits.to(torch.float32)
            teacher_score = float(cached["score"])
        else:
            teacher_logits = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
            teacher_score = 0.0

        return (
            image_tensor,
            mask_tensor,
            prompt_coords,
            ann["image_id"],
            teacher_logits,
            teacher_score,
        )


def custom_collate(batch):
    images, masks, prompts, img_ids, teacher_logits, teacher_scores = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    teacher_logits = torch.stack(teacher_logits, dim=0)
    
    prompts = torch.tensor(prompts, dtype=torch.long)
    teacher_scores = torch.tensor(teacher_scores, dtype=torch.float32)
    img_ids = list(img_ids)
    return images, masks, prompts, img_ids, teacher_logits, teacher_scores
