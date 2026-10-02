"""
Evaluate a TrOCR model on a MathWriting split.

Baseline:
  python evaluate.py --tag baseline --split-dir data/raw/mathwriting-2024/test

Fine-tuned:
  python evaluate.py --tag finetuned --model models/finetuned/best \
         --split-dir data/raw/mathwriting-2024/test

Use the SAME --limit, --seed and --pad for both so the comparison is fair.
Raw model output is scored: no overline removal, no fraction splitting,
no other demo post-processing.
"""

import argparse
import csv
import glob
import json
import os
import random
import time

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from common import parse_inkml, render_strokes, compute_metrics, \
    normalize_latex, edit_distance


def collect_files(split_dir, limit, seed):
    files = sorted(glob.glob(os.path.join(split_dir, "*.inkml")))
    random.Random(seed).shuffle(files)
    return files[:limit] if limit else files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fhswf/TrOCR_Math_handwritten")
    ap.add_argument("--split-dir",
                    default=os.path.join("data", "raw", "mathwriting-2024", "test"))
    ap.add_argument("--limit", type=int, default=0, help="0 = whole split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", choices=["square", "stretch"], default="square")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model:", args.model, "on", device)
    processor = TrOCRProcessor.from_pretrained(
        args.model, local_files_only=args.local_only)
    model = VisionEncoderDecoderModel.from_pretrained(
        args.model, local_files_only=args.local_only).to(device)
    model.eval()

    files = collect_files(args.split_dir, args.limit, args.seed)
    print("Files to evaluate:", len(files))

    rows = []
    skipped = 0
    start = time.time()

    for b in range(0, len(files), args.batch_size):
        batch_files = files[b:b + args.batch_size]
        images, refs, names = [], [], []

        for f in batch_files:
            try:
                strokes, label = parse_inkml(f)
                if not strokes or not label:
                    raise ValueError("no strokes or label")
                images.append(render_strokes(strokes, pad_mode=args.pad))
                refs.append(label)
                names.append(os.path.basename(f))
            except Exception as e:
                skipped += 1
                print("Skipped", f, "-", e)

        if not images:
            continue

        pixel_values = processor(
            images=images, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                num_beams=args.beams,
                max_new_tokens=args.max_new_tokens
            )
        preds = processor.batch_decode(ids, skip_special_tokens=True)

        for name, ref, pred in zip(names, refs, preds):
            p, r = normalize_latex(pred), normalize_latex(ref)
            rows.append({
                "file": name,
                "reference": ref,
                "prediction": pred,
                "exact": int(p == r),
                "cer": round(edit_distance(p, r) / max(len(r), 1), 4),
            })

        print(f"  {len(rows)}/{len(files)} done", end="\r")

    print()
    metrics = compute_metrics(
        [r["prediction"] for r in rows],
        [r["reference"] for r in rows]
    )
    metrics.update({
        "tag": args.tag,
        "model": args.model,
        "split_dir": args.split_dir,
        "pad": args.pad,
        "beams": args.beams,
        "seed": args.seed,
        "skipped": skipped,
        "seconds": round(time.time() - start, 1),
    })

    csv_path = os.path.join(args.out_dir, f"{args.tag}_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["file", "reference", "prediction", "exact", "cer"])
        w.writeheader()
        w.writerows(rows)

    json_path = os.path.join(args.out_dir, f"{args.tag}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()