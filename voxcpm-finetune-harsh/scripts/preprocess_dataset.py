import argparse
import json
import logging
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voxcpm.preprocess")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess a HuggingFace dataset for VoxCPM fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Dataset (HuggingFace)")
    g.add_argument(
        "--dataset", required=True,
        help="HF dataset repo ID, e.g. 'mozilla-foundation/common_voice_17_0'",
    )
    g.add_argument("--dataset_config", default=None,
                   help="Config / language code, e.g. 'hi' or 'en'")
    g.add_argument("--dataset_split", default="train",
                   help="Split to load, e.g. 'train', 'train+validation'")
    g.add_argument("--text_column", default="sentence",
                   help="Column containing the transcript text")
    g.add_argument("--audio_column", default="audio",
                   help="Column containing the audio (HF Audio or path)")
    g.add_argument("--hf_token", default=None,
                   help="HuggingFace auth token for gated / private datasets")
    g.add_argument("--max_samples", type=int, default=None,
                   help="Process at most N samples (for dry runs)")
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", required=True,
                   help="Root output directory; WAVs + manifests written here")
    g.add_argument("--audio_subdir", default="wavs",
                   help="Sub-directory for saved WAV files")
    g.add_argument("--val_split", type=float, default=0.05,
                   help="Fraction reserved for val.jsonl (0 = no validation set)")
    g = p.add_argument_group("Audio preprocessing")
    g.add_argument("--sample_rate", type=int, default=16_000,
                   help="Target sample rate (16000 for VoxCPM 2, 44100 for VoxCPM 1.5)")
    g.add_argument("--max_trailing_silence", type=float, default=0.5,
                   help="Max trailing silence to keep in seconds (docs: < 0.5 s)")
    g.add_argument("--silence_top_db", type=float, default=35.0,
                   help="Silence threshold: dB below peak RMS to treat as silence")
    g.add_argument("--no_volume_norm", action="store_true",
                   help="Disable RMS volume normalisation")
    g.add_argument("--target_rms_dbfs", type=float, default=-20.0,
                   help="Target RMS level in dBFS for normalisation")
    g = p.add_argument_group("Filtering (per official docs)")
    g.add_argument("--min_duration", type=float, default=3.0,
                   help="Reject clips shorter than N seconds (docs: sweet spot >= 3 s)")
    g.add_argument("--max_duration", type=float, default=30.0,
                   help="Reject clips longer than N seconds (docs: sweet spot <= 30 s)")
    g.add_argument("--min_snr_db", type=float, default=15.0,
                   help="Reject clips with estimated SNR below this value (0 = disabled)")
    g.add_argument("--min_text_len", type=int, default=3,
                   help="Reject transcripts shorter than N characters")
    g.add_argument("--max_text_len", type=int, default=500,
                   help="Reject transcripts longer than N characters")
    return p.parse_args()

def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality sinc resampling via torchaudio."""
    if orig_sr == target_sr:
        return wav
    import torch
    import torchaudio.functional as F
    t = torch.from_numpy(wav).float().unsqueeze(0)   # [1, T]
    t = F.resample(t, orig_sr, target_sr)
    return t.squeeze(0).numpy()


def _frame_rms(wav: np.ndarray, sr: int) -> np.ndarray:
    """Return per-frame RMS array (20 ms frames, 10 ms hop)."""
    frame_len = int(sr * 0.020)
    hop_len   = int(sr * 0.010)
    frames = [
        wav[i : i + frame_len]
        for i in range(0, max(0, len(wav) - frame_len + 1), hop_len)
    ]
    if not frames:
        return np.array([1e-12])
    return np.array([np.sqrt(np.mean(f ** 2) + 1e-12) for f in frames])


def trim_trailing_silence(wav: np.ndarray,sr: int,max_trailing_s: float = 0.5,top_db: float = 35.0,) -> np.ndarray:
    hop_len   = int(sr * 0.010)
    frame_len = int(sr * 0.020)

    rms       = _frame_rms(wav, sr)
    peak_rms  = rms.max()
    threshold = peak_rms * (10.0 ** (-top_db / 20.0))

    active = np.where(rms > threshold)[0]
    if len(active) == 0:
        return wav[:min(len(wav), int(sr * 0.1))]  # near-silent → stub

    last_active_sample = int(active[-1]) * hop_len + frame_len
    cut = min(last_active_sample + int(max_trailing_s * sr), len(wav))
    return wav[:cut]


def normalize_volume(wav: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms        = np.sqrt(np.mean(wav ** 2) + 1e-12)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain       = min(target_rms / rms, 100.0)
    return np.clip(wav * gain, -1.0, 1.0)


def estimate_snr(wav: np.ndarray, sr: int) -> float:
    rms = np.sort(_frame_rms(wav, sr))
    n   = len(rms)
    if n < 5:
        return float("inf")

    noise_rms  = float(np.median(rms[: max(1, n // 5)]))
    signal_rms = float(np.median(rms[max(0, 4 * n // 5) :]))

    if noise_rms < 1e-10:
        return float("inf")
    return 20.0 * math.log10(signal_rms / noise_rms + 1e-12)


_ABBREV_MAP: Dict[str, str] = {
    r"\bMr\.": "Mister",
    r"\bMrs\.": "Missus",
    r"\bMs\.": "Miss",
    r"\bDr\.": "Doctor",
    r"\bProf\.": "Professor",
    r"\bSt\.": "Saint",
    r"\bAve\.": "Avenue",
    r"\bBlvd\.": "Boulevard",
    r"\bvs\.": "versus",
    r"\betc\.": "et cetera",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
}

# Compile once
_ABBREV_PATTERNS = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in _ABBREV_MAP.items()]

# Characters that should never appear in a TTS transcript
_GARBAGE_RE = re.compile(r"[^\w\s\'\"\,\.\!\?\;\:\-\(\)\[\]\/\&\%\$\#\@\+\=]", re.UNICODE)


def preprocess_transcript(text: str) -> str:
    """
    Full transcript preprocessing pipeline.

    Steps
    -----
    1.  Unicode NFC normalisation  (resolves composed vs decomposed accents)
    2.  Strip leading / trailing whitespace
    3.  Collapse multiple internal spaces / tabs / newlines → single space
    4.  Expand common abbreviations  (Mr. → Mister, Dr. → Doctor, …)
    5.  Normalise quotation marks    (" " → ", ' ' → ')
    6.  Normalise dashes             (em/en dash → hyphen)
    7.  Normalise ellipsis           (… or ...) → "..."
    8.  Remove repeated punctuation  (!! → !)
    9.  Strip leading / trailing punctuation artefacts
    10. Final whitespace collapse
    """

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2 & 3. Strip + collapse whitespace
    text = " ".join(text.split())

    # 4. Abbreviation expansion
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)

    # 5. Normalise smart / curly quotes → straight quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # " "
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # ' '
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')   # « »
    text = text.replace("\u2039", "'").replace("\u203a", "'")   # ‹ ›

    # 6. Normalise dashes
    text = text.replace("\u2014", "-").replace("\u2013", "-")   # — –
    text = text.replace("\u2012", "-").replace("\u2010", "-")   # ‒ ‐

    # 7. Normalise ellipsis
    text = text.replace("\u2026", "...")                         # …
    text = re.sub(r"\.{4,}", "...", text)                        # more than 3 dots → ...

    # 8. Collapse repeated punctuation (!! → !, ?? → ?)
    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = re.sub(r",{2,}", ",", text)

    # 9. Remove leading/trailing punctuation that makes no sense for TTS
    text = text.strip(" \t\n.,;:")

    # 10. Final whitespace pass
    text = " ".join(text.split())

    return text


def is_valid_transcript(text: str, min_len: int, max_len: int) -> Tuple[bool, str]:
    """Return (is_valid, rejection_reason)."""
    if len(text) < min_len:
        return False, f"too short ({len(text)} < {min_len})"
    if len(text) > max_len:
        return False, f"too long ({len(text)} > {max_len})"
    # Must contain at least one alphabetic character
    if not any(c.isalpha() for c in text):
        return False, "no alphabetic characters"
    # Reject if > 40 % is non-alphanumeric (likely OCR garbage)
    non_alnum = sum(1 for c in text if not c.isalnum() and c != " ")
    if len(text) > 0 and non_alnum / len(text) > 0.40:
        return False, f"too many non-alphanumeric chars ({non_alnum}/{len(text)})"
    return True, ""


def process_sample(
    idx: int,
    audio_array: np.ndarray,
    audio_sr: int,
    raw_text: str,
    wav_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict], str]:
    """
    Run the full preprocessing pipeline on one (audio, text) pair.

    Returns
    -------
    (manifest_dict, "")     → accepted
    (None, reason_string)   → rejected
    """

    # ── Transcript ───────────────────────────────────────────────────────
    text = preprocess_transcript(raw_text)
    valid, reason = is_valid_transcript(text, args.min_text_len, args.max_text_len)
    if not valid:
        return None, f"[text] {reason}"

    # ── Audio: float32 mono ───────────────────────────────────────────────
    wav = audio_array.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)   # stereo → mono

    # ── Resample ──────────────────────────────────────────────────────────
    wav = resample(wav, audio_sr, args.sample_rate)

    # ── Trim trailing silence ─────────────────────────────────────────────
    wav = trim_trailing_silence(
        wav,
        sr=args.sample_rate,
        max_trailing_s=args.max_trailing_silence,
        top_db=args.silence_top_db,
    )

    # ── Duration filter (evaluated *after* trimming) ──────────────────────
    duration = len(wav) / args.sample_rate
    if duration < args.min_duration:
        return None, f"[duration] {duration:.2f}s < min {args.min_duration}s"
    if duration > args.max_duration:
        return None, f"[duration] {duration:.2f}s > max {args.max_duration}s"

    # ── SNR filter ────────────────────────────────────────────────────────
    if args.min_snr_db > 0:
        snr = estimate_snr(wav, args.sample_rate)
        if snr < args.min_snr_db:
            return None, f"[snr] {snr:.1f} dB < {args.min_snr_db} dB"

    # ── Volume normalisation ──────────────────────────────────────────────
    if not args.no_volume_norm:
        wav = normalize_volume(wav, target_dbfs=args.target_rms_dbfs)

    # ── Save WAV ──────────────────────────────────────────────────────────
    wav_path = wav_dir / f"{idx:08d}.wav"
    sf.write(str(wav_path), wav, args.sample_rate, subtype="PCM_16")

    return {
        "audio": str(wav_path),
        "text": text,
        "duration": round(duration, 4),
    }, ""


class Stats:
    def __init__(self):
        self.total: int = 0
        self.accepted: int = 0
        self.rejected: Dict[str, int] = {}
        self.durations: List[float] = []       # accepted durations in seconds
        self.text_lengths: List[int] = []      # accepted transcript char counts
        self.snr_values: List[float] = []      # SNR of accepted samples

    def record_reject(self, reason: str):
        self.total += 1
        tag = reason.split("]")[0].lstrip("[") if "]" in reason else "other"
        self.rejected[tag] = self.rejected.get(tag, 0) + 1

    def record_accept(self, duration: float, text_len: int, snr: float = float("nan")):
        self.total += 1
        self.accepted += 1
        self.durations.append(duration)
        self.text_lengths.append(text_len)
        if not math.isnan(snr):
            self.snr_values.append(snr)

    # ── Helper: compute percentile safely ────────────────────────────────
    @staticmethod
    def _pct(arr: List[float], q: float) -> float:
        return float(np.percentile(arr, q)) if arr else 0.0

    # ── Report ────────────────────────────────────────────────────────────
    def report(self, train_n: int, val_n: int, args: argparse.Namespace) -> str:
        dur  = self.durations
        tlen = self.text_lengths
        snr  = self.snr_values

        total_s   = sum(dur)
        total_h   = total_s / 3600.0
        avg_dur   = total_s / len(dur) if dur else 0.0
        median_dur = self._pct(dur, 50)
        p10_dur   = self._pct(dur, 10)
        p90_dur   = self._pct(dur, 90)

        avg_tlen  = sum(tlen) / len(tlen) if tlen else 0
        med_tlen  = self._pct(tlen, 50)

        lines = []
        W = 60

        def hdr(title: str):
            lines.append("")
            lines.append("━" * W)
            lines.append(f"  {title}")
            lines.append("━" * W)

        def row(label: str, value: str):
            lines.append(f"  {label:<32} {value}")

        def sep():
            lines.append("  " + "─" * (W - 2))

        # ── Overview ─────────────────────────────────────────────────────
        hdr("📊  Dataset Overview")
        row("Dataset",          args.dataset)
        row("Config / split",   f"{args.dataset_config or '—'} / {args.dataset_split}")
        row("Total processed",  f"{self.total:,}")
        row("Accepted",         f"{self.accepted:,}  ({self.accepted/max(1,self.total)*100:.1f}%)")
        row("Rejected",         f"{self.total - self.accepted:,}  ({(self.total-self.accepted)/max(1,self.total)*100:.1f}%)")

        # ── Duration statistics ───────────────────────────────────────────
        hdr("⏱   Duration Statistics  (accepted clips)")
        row("Total audio",       f"{total_h:.2f} h  ({total_s:,.0f} s)")
        row("Average per clip",  f"{avg_dur:.2f} s")
        row("Median per clip",   f"{median_dur:.2f} s")
        row("10th percentile",   f"{p10_dur:.2f} s")
        row("90th percentile",   f"{p90_dur:.2f} s")
        row("Min / Max",         f"{min(dur):.2f} s  /  {max(dur):.2f} s" if dur else "—")

        # Duration histogram (7 buckets)
        if dur:
            sep()
            lines.append("  Duration distribution:")
            bucket_edges = [0, 3, 5, 8, 12, 20, 30, float("inf")]
            bucket_labels = ["<3s", "3–5s", "5–8s", "8–12s", "12–20s", "20–30s", ">30s"]
            counts = [0] * len(bucket_labels)
            for d in dur:
                for bi, (lo, hi) in enumerate(zip(bucket_edges[:-1], bucket_edges[1:])):
                    if lo <= d < hi:
                        counts[bi] += 1
                        break
            bar_max = max(counts) if counts else 1
            for label, cnt in zip(bucket_labels, counts):
                bar_w = int(cnt / bar_max * 25) if bar_max > 0 else 0
                pct   = cnt / len(dur) * 100
                lines.append(f"    {label:>7}  {'█' * bar_w:<25}  {cnt:>5,}  ({pct:5.1f}%)")

        # ── Transcript statistics ─────────────────────────────────────────
        hdr("📝  Transcript Statistics  (accepted)")
        row("Avg length (chars)",   f"{avg_tlen:.1f}")
        row("Median length (chars)", f"{med_tlen:.1f}")
        row("Min / Max (chars)",     f"{min(tlen)} / {max(tlen)}" if tlen else "—")

        # ── SNR statistics ────────────────────────────────────────────────
        if snr:
            hdr("🔊  SNR Statistics  (accepted samples)")
            row("Average SNR",     f"{np.mean(snr):.1f} dB")
            row("Median SNR",      f"{np.median(snr):.1f} dB")
            row("Min SNR",         f"{min(snr):.1f} dB")
            row("5th percentile",  f"{self._pct(snr, 5):.1f} dB")

        # ── Rejection breakdown ───────────────────────────────────────────
        if self.rejected:
            hdr("🚫  Rejection Breakdown")
            for tag, cnt in sorted(self.rejected.items(), key=lambda x: -x[1]):
                pct = cnt / self.total * 100
                bar = "█" * int(pct / 2)
                lines.append(f"  [{tag:<12}]  {bar:<25}  {cnt:>6,}  ({pct:5.1f}%)")

        # ── Train / val split ─────────────────────────────────────────────
        hdr("📂  Output Manifest")
        train_h = sum(d for d in dur[:train_n])  # approximate; order may differ
        row("train.jsonl  samples",  f"{train_n:,}")
        row("val.jsonl    samples",  f"{val_n:,}")
        row("Target sample rate",    f"{args.sample_rate:,} Hz")
        row("Volume normalisation",  "disabled" if args.no_volume_norm else f"RMS → {args.target_rms_dbfs} dBFS")
        row("Silence threshold",     f"{args.silence_top_db} dB  /  tail ≤ {args.max_trailing_silence}s")
        row("SNR filter",            f"≥ {args.min_snr_db} dB" if args.min_snr_db > 0 else "disabled")

        lines.append("")
        lines.append("━" * W)
        lines.append("")

        return "\n".join(lines)

def main():
    args = parse_args()

    # Setup output dirs
    out_dir = Path(args.output_dir)
    wav_dir = out_dir / args.audio_subdir
    wav_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory : {out_dir.resolve()}")
    logger.info(f"WAV sub-dir      : {wav_dir.resolve()}")

    # ── Load dataset ─────────────────────────────────────────────────────
    logger.info(
        f"Loading dataset  : {args.dataset}  "
        f"config={args.dataset_config or '—'}  split={args.dataset_split}"
    )
    from datasets import load_dataset, Audio

    load_kwargs: Dict = dict(
        path=args.dataset,
        split=args.dataset_split,
        trust_remote_code=True,
    )
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    ds = load_dataset(**load_kwargs)

    # Validate columns
    for col, name in [(args.audio_column, "audio"), (args.text_column, "text")]:
        if col not in ds.column_names:
            raise ValueError(
                f"{name.capitalize()} column '{col}' not found. "
                f"Available columns: {ds.column_names}"
            )

    # Cast audio column (keep original SR; we resample ourselves for quality control)
    ds = ds.cast_column(args.audio_column, Audio(sampling_rate=None))

    total = min(len(ds), args.max_samples) if args.max_samples else len(ds)
    logger.info(f"Processing {total:,} samples ...")
    logger.info(f"Transcript preprocessing : enabled (NFC, abbrev expansion, punct normalisation)")
    logger.info(f"Trailing silence trim    : ≤ {args.max_trailing_silence}s  (threshold: {args.silence_top_db} dB)")
    logger.info(f"Duration filter          : {args.min_duration}–{args.max_duration}s  [per official docs: 3–30s sweet spot]")
    logger.info(f"SNR filter               : {'≥ ' + str(args.min_snr_db) + ' dB' if args.min_snr_db > 0 else 'disabled'}")
    logger.info(f"Volume norm              : {'disabled' if args.no_volume_norm else str(args.target_rms_dbfs) + ' dBFS RMS'}")

    # ── Process ───────────────────────────────────────────────────────────
    stats = Stats()
    manifest_entries: List[Dict] = []

    try:
        from tqdm import tqdm
        it = tqdm(range(total), unit="sample", dynamic_ncols=True, desc="Processing")
    except ImportError:
        it = range(total)

    for i in it:
        row = ds[i]

        # Extract audio
        audio_data = row[args.audio_column]
        if not isinstance(audio_data, dict) or "array" not in audio_data:
            stats.record_reject("[load] unexpected audio format")
            continue

        audio_array = np.array(audio_data["array"], dtype=np.float32)
        audio_sr    = int(audio_data["sampling_rate"])

        # Extract raw text
        raw_text = str(row.get(args.text_column, "") or "").strip()
        if not raw_text:
            stats.record_reject("[text] empty transcript")
            continue

        # Estimate SNR upfront so we can include it in stats
        wav_mono = audio_array.astype(np.float32)
        if wav_mono.ndim > 1:
            wav_mono = wav_mono.mean(axis=-1)
        wav_resampled = resample(wav_mono, audio_sr, args.sample_rate)
        snr_val = estimate_snr(wav_resampled, args.sample_rate) if args.min_snr_db > 0 else float("nan")

        # Full pipeline
        entry, reason = process_sample(
            idx=i,
            audio_array=audio_array,
            audio_sr=audio_sr,
            raw_text=raw_text,
            wav_dir=wav_dir,
            args=args,
        )

        if entry is None:
            stats.record_reject(reason)
        else:
            stats.record_accept(
                duration=entry["duration"],
                text_len=len(entry["text"]),
                snr=snr_val,
            )
            manifest_entries.append(entry)

    if not manifest_entries:
        logger.error(
            "No samples passed the filters!\n"
            "  → Try relaxing: --min_snr_db, --min_duration, --max_duration\n"
            "  → Or check that --text_column and --audio_column are correct."
        )
        sys.exit(1)

    # ── Train / val split ─────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    shuffled = rng.permutation(len(manifest_entries)).tolist()

    n_val = int(len(manifest_entries) * args.val_split) if args.val_split > 0 else 0
    n_val = max(0, min(n_val, len(manifest_entries) - 1))

    val_entries   = [manifest_entries[i] for i in shuffled[:n_val]]
    train_entries = [manifest_entries[i] for i in shuffled[n_val:]]

    # ── Write manifests ───────────────────────────────────────────────────
    def write_jsonl(path: Path, entries: List[Dict]):
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    write_jsonl(train_path, train_entries)
    if val_entries:
        write_jsonl(val_path, val_entries)

    # ── Full statistics report ────────────────────────────────────────────
    report = stats.report(
        train_n=len(train_entries),
        val_n=len(val_entries),
        args=args,
    )
    print(report)

    # ── Config snippet ────────────────────────────────────────────────────
    print("  📄  Paste into your training YAML:")
    print()
    print(f"      pretrained_path: pretrained_models/VoxCPM2")
    print(f"      train_manifest:  {train_path.resolve()}")
    if val_entries:
        print(f"      val_manifest:    {val_path.resolve()}")
    print(f"      sample_rate:     {args.sample_rate}")
    print()

    logger.info("Done.")


if __name__ == "__main__":
    main()
