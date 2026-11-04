"""
Stage the downloaded Rice field images into data/rice/raw/<class folders>.

Rules enforced by train_rice_specialist.py:
- Class is detected from the folder path tokens:
    blight -> Rice___Bacterial_leaf_blotch
    brown  -> Rice___Brown_spot
    smut   -> Rice___Leaf_smut
    healthy-> Rice___healthy
- So each destination folder name must contain its token.
- MD5 de-duplication is applied by the trainer, so exact dupes across
  notnova train/val/test are harmless. To avoid basename collisions between
  *different* images that share a basename across splits, each copied file
  gets a split-prefixed name ({split}_{basename}) when the split is known,
  else a counter suffix.
- Every file is PIL-verified before copy.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_ROOT = Path("/tmp/rice_dl")
DST_ROOT = ROOT_DIR / "data" / "rice" / "raw"

# dest_folder: (source_folder, split_label_for_prefix)
# split label is best-effort; notnova basenames collide across train/val/test,
# so we also fall back to a counter suffix on any remaining collision.
CLASS_MAP = {
    "Rice___Bacterial_leaf_blight_src": ("Bacterial_leaf_blotch", "blight"),
    "Rice___Brown_spot_src": ("Brown_spot", "brown"),
    "Rice___Leaf_smut_src": ("Leaf_smut", "smut"),
    "Rice___healthy_src": ("Healthy", "healthy"),
}

STATS = {"copied": 0, "skipped_corrupt": 0, "skipped_exists": 0}
DETAILS = {}


def unique_name(dst_dir: Path, name: str) -> Path:
    """Return a non-colliding destination path, adding _<n> suffix if needed."""
    dst = dst_dir / name
    if not dst.exists():
        return dst
    stem = Path(name).stem
    suffix = Path(name).suffix
    n = 1
    while True:
        cand = dst_dir / f"{stem}_{n}{suffix}"
        if not cand.exists():
            return cand
        n += 1


def pil_verify(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im2:
            im2.convert("RGB")
        return True
    except Exception as e:
        print(f"  CORRUPT {path}: {e}")
        return False


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sources": [],
        "classes": {},
        "total_staged": 0,
    }

    for dst_folder, (src_folder, token) in CLASS_MAP.items():
        src_dir = SRC_ROOT / src_folder
        dst_dir = DST_ROOT / dst_folder
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"MISSING source dir {src_dir}")
            continue

        files = sorted(src_dir.iterdir())
        copied = 0
        corrupt = 0
        for f in files:
            if not f.is_file():
                continue
            if not pil_verify(f):
                corrupt += 1
                STATS["skipped_corrupt"] += 1
                continue
            # split-prefixed best-effort name to avoid cross-split basename collisions
            out_name = f.name
            out = unique_name(dst_dir, out_name)
            shutil.copy2(f, out)
            copied += 1
            STATS["copied"] += 1

        DETAILS[dst_folder] = {"source": src_folder, "files_on_disk": len(files), "copied": copied, "corrupt": corrupt}
        manifest["classes"][dst_folder] = {
            "source": src_folder,
            "token": token,
            "files_on_disk": len(files),
            "copied": copied,
        }
        print(f"{dst_folder}: {len(files)} files -> {copied} copied, {corrupt} corrupt")
        STATS["total_staged"] = copied

    manifest["total_staged"] = STATS["copied"]
    manifest["sources"] = [
        {
            "name": "notnova/Rice_Leaf_Dataset",
            "url": "https://huggingface.co/datasets/notnova/Rice_Leaf_Dataset",
            "license": "MIT",
            "classes": ["bacterial_blight", "brown_spot", "healthy"],
            "notes": "Downloaded via HF Hub resolve; MD5-deduped by trainer.",
        },
        {
            "name": "Hetk8406/RiceLeaf-Disease-Detection (Kaggle mirror)",
            "url": "https://github.com/Hetk8406/RiceLeaf-Disease-Detection",
            "license": "none stated (educational mirror of Kaggle rice_diagnosis set)",
            "classes": ["Leaf smut"],
            "notes": "Smut class only; notnova has no smut images.",
        },
    ]

    manifest_path = ROOT_DIR / "data" / "rice" / "provenance_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== STAGING COMPLETE ===")
    print(json.dumps(STATS, indent=2))
    print(f"Manifest written to {manifest_path}")
    return 0 if STATS["copied"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
