import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voxcpm.merge")

W = 60  # report width


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge multiple VoxCPM preprocessed dataset directories into one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets", nargs="+", required=True,
        help=(
            "One or more dataset directories (output of preprocess_dataset.py). "
            "Each must contain train.jsonl and optionally val.jsonl."
        ),
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Where to write the merged train.jsonl and val.jsonl.",
    )
    p.add_argument(
        "--val_split", type=float, default=0.05,
        help=(
            "Fraction of the merged pool to use for val.jsonl. "
            "Set to 0 to skip validation set creation."
        ),
    )
    p.add_argument(
        "--max_samples_per_dataset", nargs="*", type=int, default=None,
        help=(
            "Optional per-dataset sample cap (same order as --datasets). "
            "Use this to balance datasets of very different sizes. "
            "Example: --max_samples_per_dataset 30000 30000 5000"
        ),
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling.",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"  Skipping bad JSON at {path}:{lineno} — {e}")
    return entries

def write_jsonl(path: Path, entries: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def dataset_stats(entries: List[Dict], name: str) -> Dict:
    durations = [e.get("duration", 0.0) for e in entries]
    total_s   = sum(durations)
    avg_dur   = total_s / len(durations) if durations else 0.0
    return {
        "name":    name,
        "samples": len(entries),
        "total_h": total_s / 3600.0,
        "total_s": total_s,
        "avg_dur": avg_dur,
    }

def print_report(
    per_dataset: List[Dict],
    merged_total: int,
    train_n: int,
    val_n: int,
    merged_dur_h: float,
    output_dir: Path,
):
    def hdr(title):
        print("")
        print("━" * W)
        print(f"  {title}")
        print("━" * W)

    def row(label, value):
        print(f"  {label:<32} {value}")

    def sep():
        print("  " + "─" * (W - 2))

    hdr("📦  Per-Dataset Breakdown")
    row("Dataset", f"{'Samples':>10}  {'Hours':>8}  {'Avg dur':>8}")
    sep()
    for ds in per_dataset:
        row(
            Path(ds["name"]).name,
            f"{ds['samples']:>10,}  {ds['total_h']:>7.2f}h  {ds['avg_dur']:>7.2f}s",
        )
    sep()
    row("TOTAL", f"{merged_total:>10,}  {merged_dur_h:>7.2f}h")

    hdr("📂  Merged Output")
    row("train.jsonl  samples", f"{train_n:,}")
    row("val.jsonl    samples", f"{val_n:,}")
    row("Total audio  (train)", f"≈ {merged_dur_h * (train_n / max(1, merged_total)):.2f} h")
    row("Output dir", str(output_dir.resolve()))

    # Recommendation
    hdr("💡  Fine-Tuning Recommendation")
    if merged_dur_h < 50:
        rec = "LoRA  (data < 50 h — LoRA is ideal)"
    elif merged_dur_h < 500:
        rec = "LoRA  (data < 500 h — LoRA recommended; try full FT after)"
    else:
        rec = "Full Fine-Tuning  (data >= 500 h — consider full FT)"
    row("Approach", rec)

    print("")
    print("━" * W)
    print("")
    print("  📄  Paste into your training YAML:")
    print()
    print(f"      pretrained_path: pretrained_models/VoxCPM2")
    print(f"      train_manifest:  {(output_dir / 'train.jsonl').resolve()}")
    if val_n > 0:
        print(f"      val_manifest:    {(output_dir / 'val.jsonl').resolve()}")
    print(f"      sample_rate:     16000")
    print()


def main():
    args = parse_args()

    dataset_dirs = [Path(d) for d in args.datasets]
    output_dir   = Path(args.output_dir)

    # Validate --max_samples_per_dataset length
    caps: List[Optional[int]] = [None] * len(dataset_dirs)
    if args.max_samples_per_dataset is not None:
        if len(args.max_samples_per_dataset) != len(dataset_dirs):
            logger.error(
                f"--max_samples_per_dataset has {len(args.max_samples_per_dataset)} values "
                f"but --datasets has {len(dataset_dirs)}. They must match."
            )
            sys.exit(1)
        caps = args.max_samples_per_dataset

    rng = np.random.default_rng(seed=args.seed)

    all_entries:  List[Dict] = []
    per_dataset_stats: List[Dict] = []

    logger.info(f"Merging {len(dataset_dirs)} dataset(s) ...")

    for ds_dir, cap in zip(dataset_dirs, caps):
        # Collect all entries: prefer separate train+val, fall back to train-only
        train_path = ds_dir / "train.jsonl"
        val_path   = ds_dir / "val.jsonl"

        entries: List[Dict] = []

        if train_path.exists():
            t = read_jsonl(train_path)
            logger.info(f"  {ds_dir.name}/train.jsonl  →  {len(t):,} entries")
            entries.extend(t)
        else:
            logger.warning(f"  {ds_dir}: train.jsonl not found, skipping.")

        if val_path.exists():
            v = read_jsonl(val_path)
            logger.info(f"  {ds_dir.name}/val.jsonl    →  {len(v):,} entries  (pooled into merge)")
            entries.extend(v)

        if not entries:
            logger.warning(f"  {ds_dir}: no entries found, skipping.")
            continue

        # Apply per-dataset cap
        if cap is not None and cap < len(entries):
            idx = rng.choice(len(entries), size=cap, replace=False)
            entries = [entries[i] for i in idx]
            logger.info(f"  {ds_dir.name}: capped to {cap:,} samples")

        per_dataset_stats.append(dataset_stats(entries, str(ds_dir)))
        all_entries.extend(entries)

    if not all_entries:
        logger.error("No entries collected from any dataset. Check your --datasets paths.")
        sys.exit(1)

    total = len(all_entries)
    total_dur_h = sum(e.get("duration", 0.0) for e in all_entries) / 3600.0
    logger.info(f"\nTotal pooled: {total:,} samples  ({total_dur_h:.2f} h)")

    # Global shuffle
    logger.info("Shuffling ...")
    indices = rng.permutation(total).tolist()
    all_entries = [all_entries[i] for i in indices]

    # Train / val split
    n_val = int(total * args.val_split) if args.val_split > 0 else 0
    n_val = max(0, min(n_val, total - 1))

    val_entries   = all_entries[:n_val]
    train_entries = all_entries[n_val:]

    # Write
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_entries)
    logger.info(f"Wrote {len(train_entries):,} entries → {output_dir / 'train.jsonl'}")

    if val_entries:
        write_jsonl(output_dir / "val.jsonl", val_entries)
        logger.info(f"Wrote {len(val_entries):,} entries → {output_dir / 'val.jsonl'}")

    # Report
    print_report(
        per_dataset=per_dataset_stats,
        merged_total=total,
        train_n=len(train_entries),
        val_n=len(val_entries),
        merged_dur_h=total_dur_h,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
