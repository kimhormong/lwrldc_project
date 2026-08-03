import shutil
import math
import random
from pathlib import Path
from typing import Dict

# Explicit Pillow import
import helper_utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchvision.io as io
from torchvision.transforms.v2 import functional as F

# =====================================================================
# PART 1: 80/10/10 SPLITTING LOGIC (TRAIN / VAL / TEST)
# =====================================================================

def split_class_images_proportionally(
    source_dir: Path,
    splits_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Splits an image dataset into train/val/test subdirectories by calculating
    an 80/10/10 split INDEPENDENTLY for every class subfolder.
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    
    random.seed(seed)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

    class_folders = [f for f in source_dir.iterdir() if f.is_dir()]
    if not class_folders:
        raise ValueError(f"No class directories found inside '{source_dir}'")

    print(f"--> Found {len(class_folders)} classes. Processing 80/10/10 Train/Val/Test splits...")

    summary_stats: Dict[str, Dict[str, int]] = {}

    for class_folder in class_folders:
        class_name = class_folder.name
        
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in valid_extensions]
        random.shuffle(images)
        
        total_imgs = len(images)
        if total_imgs == 0:
            print(f"Skipping empty class directory: {class_name}")
            continue

        # Compute exact image counts per split
        train_count = math.floor(total_imgs * train_ratio)
        val_count = math.floor(total_imgs * val_ratio)
        # Allocate remaining samples to test to guarantee exact total count match
        test_count = total_imgs - train_count - val_count

        splits = {
            "train": images[:train_count],
            "val": images[train_count : train_count + val_count],
            "test": images[train_count + val_count:]
        }

        summary_stats[class_name] = {
            "total": total_imgs,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"])
        }

        # Copy files into temp/splits directory
        for split_name, file_list in splits.items():
            dest_dir = splits_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in file_list:
                dest_file = dest_dir / img_path.name
                shutil.copy2(img_path, dest_file)

    # Print split summary table
    print(f"\n{'Class Name':<20} | {'Total':<7} | {'Train':<7} | {'Val':<7} | {'Test':<7}")
    print("-" * 60)
    for cls_name, stats in summary_stats.items():
        print(f"{cls_name:<20} | {stats['total']:<7} | {stats['train']:<7} | {stats['val']:<7} | {stats['test']:<7}")
    print("\n[+] Splitting complete.")


# =====================================================================
# PART 2: DATASET & RESIZING VIA PILLOW
# =====================================================================

class SimpleJpgResizerDataset(Dataset):
    def __init__(self, src_dir: Path, target_size=(224, 224)):
        self.src_dir = src_dir
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
        self.image_paths = [
            p for p in self.src_dir.rglob("*") if p.suffix.lower() in valid_exts
        ]
        
        self.target_size = target_size
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # 1. Read image directly into a PyTorch tensor using a PyTorch Tensor
            img = io.read_image(str(img_path), mode=io.ImageReadMode.RGB)  # Shape: [C, H, W], dtype: uint8
            
            # 2. Resize using Pytorch functional Transforms
            resized_img = F.resize(img, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            
            return resized_img, str(img_path)
        except Exception:
            return None, str(img_path)


def resize_and_save_all_to_jpg(src_dir: Path, dst_dir: Path, target_size=(224, 224), batch_size=32, num_workers=2):
    dataset = SimpleJpgResizerDataset(src_dir, target_size=target_size)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda batch: [b for b in batch if b[0] is not None],
        shuffle=False
    )

    print(f"--> Resizing {len(dataset)} images from '{src_dir}' using PyTorch...")

    total_saved = 0
    for batch in loader:
        for resized_img, orig_path in batch:
            orig_path = Path(orig_path)
            relative_path = orig_path.relative_to(src_dir)
            
            # Save maintaining split/class_name/file.jpg layout
            save_path = (dst_dir / relative_path).with_suffix(".jpg")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            io.write_jpeg(resized_img, str(save_path), quality=100,)
            total_saved += 1

    print(f"[+] Resized and saved {total_saved} images to '{dst_dir}'")


# =====================================================================
# PART 3: PIPELINE EXECUTION
# =====================================================================

def split_then_resize_pipeline(
    raw_dataset_dir: str,
    output_dir: str,
    target_size=(224, 224),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 2,
    keep_unresized_splits: bool = False
):
    """
    1. Splits raw data into 80/10/10 train/val/test structure.
    2. Resizes all images using Pillow to target_size and saves to output_dir.
    3. Cleans up temporary split files.
    """
    raw_path = Path(raw_dataset_dir)
    final_output_path = Path(output_dir)
    temp_splits_path = Path(f"{output_dir}_temp_unresized_splits")

    try:
        # Step 1: Perform 80/10/10 Train/Val/Test split
        print("=== STEP 1: SPLITTING DATASET (80% Train / 10% Val / 10% Test) ===")
        split_class_images_proportionally(
            source_dir=raw_path,
            splits_dir=temp_splits_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )

        # Step 2: Resize dataset via Pillow & PyTorch DataLoader
        print("\n=== STEP 2: RESIZING & CONVERTING IMAGES ===")
        resize_and_save_all_to_jpg(
            src_dir=temp_splits_path,
            dst_dir=final_output_path,
            target_size=target_size,
            batch_size=batch_size,
            num_workers=num_workers
        )

        print("\n=== PIPELINE SUCCESSFUL ===")
        print(f"Final resized dataset location: {final_output_path.resolve()}")

    finally:
        if not keep_unresized_splits and temp_splits_path.exists():
            print("\nCleaning up intermediate temporary split files...")
            shutil.rmtree(temp_splits_path)


if __name__ == "__main__":
    configs = helper_utils.load_config("../configs/dataset_config.yaml")
    dataset = configs.dataset

    split_then_resize_pipeline(
        raw_dataset_dir=f"../{dataset.raw_data_dir}",           # Input raw directory
        output_dir=f"../{dataset.processed_data_dir}",  # Output directory containing train/ and test/
        target_size=(dataset.image_size, dataset.image_size),              # Target image size
        seed=dataset.seed,                             # Seed for split reproducibility
        keep_unresized_splits=False,         # Delete raw intermediate splits after resizing
        train_ratio=dataset.train_ratio, 
        val_ratio=dataset.val_ratio, 
        test_ratio=dataset.test_ratio,
        batch_size=64,
        num_workers=0,
    )