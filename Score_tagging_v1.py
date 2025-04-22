"""
Recursively tag every image under a chosen root directory with a YOLO classifier
and store a mapped tag next to each image.

Changes (22 Apr 2025)
─────────────────────
• Added `select_model_file()` to let the user choose a .pt model file via GUI.
• `main()` now calls that helper and feeds the chosen path into YOLO().
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import numpy as np


# ── I/O helpers ──────────────────────────────────────────────────────────────
def select_root_folder() -> str:
    """Open a folder‑picker and return the selected path."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select the *top‑level* folder containing images")
    if not folder:
        raise SystemExit("No folder selected — cancelled.")
    root.destroy()
    return folder


def select_model_file() -> str:
    """Open a file‑picker and return the selected YOLO .pt model path."""
    root = tk.Tk()
    root.withdraw()
    filetypes = [("PyTorch model", "*.pt"), ("All files", "*.*")]
    model_path = filedialog.askopenfilename(title="Select a YOLO model (.pt)", filetypes=filetypes)
    if not model_path:
        raise SystemExit("No model selected — cancelled.")
    root.destroy()
    return model_path


def gather_image_paths(root: str, exts: Optional[set[str]] = None) -> List[str]:
    """Recursively collect image paths whose extension is in *exts*."""
    exts = exts or {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]


# ── Image utilities ──────────────────────────────────────────────────────────
def load_and_resize(path: str, target: int = 224) -> Optional[np.ndarray]:
    """
    Read *path* with Pillow and resize so the shorter edge == *target* px
    (aspect ratio preserved). Return an RGB numpy array.
    """
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w <= h:
            new_w, new_h = target, int(h * target / w)
        else:
            new_w, new_h = int(w * target / h), target
        return np.asarray(img.resize((new_w, new_h), Image.Resampling.LANCZOS))
    except Exception as e:
        tqdm.write(f"⚠️  {Path(path).name}: {e}")
        return None


def append_tag_file(image_path: str, tag: str) -> None:
    """Create (or append to) a .txt file next to *image_path* with *tag*."""
    txt_path = Path(image_path).with_suffix(".txt")
    try:
        existing = txt_path.read_text().strip() if txt_path.exists() else ""
        txt_path.write_text(f"{existing},{tag}" if existing else tag)
    except Exception as e:
        tqdm.write(f"⚠️  Could not update {txt_path.name}: {e}")


# ── Asynchronous pipeline ────────────────────────────────────────────────────
async def classify_one(
    path: str,
    model: YOLO,
    mapping: Dict[str, str],
    sem: asyncio.Semaphore,
    pbar: tqdm,
) -> Optional[str]:
    async with sem:
        img = await asyncio.to_thread(load_and_resize, path)
        if img is None:
            pbar.update(1)
            return None

        result = await asyncio.to_thread(lambda: model(img)[0])

        if not result.probs:
            tqdm.write(f"⚠️  No prediction for {Path(path).name}")
            pbar.update(1)
            return None

        class_idx = int(result.probs.top1)
        raw_label = model.names.get(class_idx, str(class_idx))
        tag = mapping.get(raw_label)

        if tag:
            await asyncio.to_thread(append_tag_file, path, tag)
        else:
            tqdm.write(f"ℹ️  No mapping for '{raw_label}' in {Path(path).name}")

        pbar.update(1)
        return tag


async def process_directory(
    root: str,
    model: YOLO,
    mapping: Dict[str, str],
    target_size: int = 224,
    concurrency: int = 10,
) -> None:
    paths = gather_image_paths(root)
    sem = asyncio.Semaphore(concurrency)
    tally: Dict[str, int] = {}

    with tqdm(total=len(paths), desc="Tagging images") as pbar:
        tasks = [classify_one(p, model, mapping, sem, pbar) for p in paths]
        for tag in await asyncio.gather(*tasks):
            if tag:
                tally[tag] = tally.get(tag, 0) + 1

    print("\n── Tag summary ──")
    for k, v in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k:<15} {v}")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    root_dir = select_root_folder()
    model_path = select_model_file()          # NEW ➜ pick the .pt file interactively
    model = YOLO(model_path)

    custom_mapping = {
        "Top10": "masterpiece",
        "Top20": "best quality",
        "Top30": "best quality",
        "Top80": "low quality",
        "Top90": "low quality",
        "Top100": "worst quality",
    }

    asyncio.run(
        process_directory(
            root_dir,
            model,
            custom_mapping,
            target_size=224,
            concurrency=10,
        )
    )


if __name__ == "__main__":
    main()

