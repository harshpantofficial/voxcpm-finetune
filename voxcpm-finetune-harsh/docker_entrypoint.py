import os
import sys
import yaml
import json
import subprocess
from pathlib import Path


def parse_env_value(val: str):
    """Auto-cast environment variable string values to appropriate Python types."""
    val_strip = val.strip()
    if val_strip.lower() == "true":
        return True
    if val_strip.lower() == "false":
        return False
    if val_strip.lower() in ("none", "null"):
        return None
    if (val_strip.startswith("{") and val_strip.endswith("}")) or (val_strip.startswith("[") and val_strip.endswith("]")):
        try:
            return json.loads(val_strip)
        except Exception:
            pass
    try:
        if val_strip.isdigit() or (val_strip.startswith("-") and val_strip[1:].isdigit()):
            return int(val_strip)
    except ValueError:
        pass
    try:
        return float(val_strip)
    except ValueError:
        pass
    return val


def update_nested_dict(d: dict, keys: list, value):
    curr = d
    for key in keys[:-1]:
        if key not in curr or not isinstance(curr[key], dict):
            curr[key] = {}
        curr = curr[key]
    curr[keys[-1]] = value


def sanitize_dataset_name(repo_id: str) -> str:
    return repo_id.strip().replace("/", "__").replace(" ", "_")


def parse_dataset_urls(raw: str) -> list:
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def load_dataset_specs(urls: list) -> list:
    """Build per-dataset kwargs. Global env vars set the defaults; DATASET_SPECS_JSON
    (a list aligned by index with DATASET_URLS) can override any field per dataset."""
    defaults = {
        "dataset_config": os.environ.get("DATASET_CONFIG", ""),
        "dataset_split": os.environ.get("DATASET_SPLIT", "train"),
        "text_column": os.environ.get("TEXT_COLUMN", "sentence"),
        "audio_column": os.environ.get("AUDIO_COLUMN", "audio"),
        "max_samples": os.environ.get("DATASET_MAX_SAMPLES", ""),
    }

    overrides_raw = os.environ.get("DATASET_SPECS_JSON", "").strip()
    overrides = json.loads(overrides_raw) if overrides_raw else []

    specs = []
    for i, repo_id in enumerate(urls):
        spec = {**defaults, "dataset": repo_id}
        if i < len(overrides):
            spec.update(overrides[i])
        specs.append(spec)
    return specs


def run_preprocess(spec: dict, output_dir: Path, hf_token: str):
    cmd = [
        sys.executable, "scripts/preprocess_dataset.py",
        "--dataset", spec["dataset"],
        "--dataset_split", spec["dataset_split"],
        "--text_column", spec["text_column"],
        "--audio_column", spec["audio_column"],
        "--output_dir", str(output_dir),
    ]
    if spec.get("dataset_config"):
        cmd += ["--dataset_config", str(spec["dataset_config"])]
    if spec.get("max_samples"):
        cmd += ["--max_samples", str(spec["max_samples"])]
    if hf_token:
        cmd += ["--hf_token", hf_token]

    print(f"[VoxCPM Docker] Preprocessing dataset: {spec['dataset']} -> {output_dir}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Preprocessing failed for dataset '{spec['dataset']}'")


def run_merge(dataset_dirs: list, output_dir: Path, val_split: float):
    cmd = [
        sys.executable, "scripts/merge_manifests.py",
        "--datasets", *[str(d) for d in dataset_dirs],
        "--output_dir", str(output_dir),
        "--val_split", str(val_split),
    ]
    print(f"[VoxCPM Docker] Merging {len(dataset_dirs)} dataset(s) -> {output_dir}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError("Manifest merge failed")


def manifest_hours(*jsonl_paths: Path) -> float:
    total_seconds = 0.0
    for path in jsonl_paths:
        if not path or not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_seconds += json.loads(line).get("duration", 0.0)
    return total_seconds / 3600.0


def run_data_pipeline() -> dict:
    """Downloads, preprocesses, and merges one or more HF datasets. Returns the
    resolved train/val manifest paths plus the actual merged dataset hours."""
    urls = parse_dataset_urls(os.environ.get("DATASET_URLS", ""))
    if not urls:
        return {}

    hf_token = os.environ.get("HF_TOKEN", "")
    data_root = Path(os.environ.get("DATA_ROOT", "data"))
    raw_root = data_root / "raw"
    merged_dir = data_root / "merged"
    val_split = float(os.environ.get("DATASET_VAL_SPLIT", "0.05"))

    specs = load_dataset_specs(urls)
    dataset_dirs = []
    for spec in specs:
        out_dir = raw_root / sanitize_dataset_name(spec["dataset"])
        run_preprocess(spec, out_dir, hf_token)
        dataset_dirs.append(out_dir)

    run_merge(dataset_dirs, merged_dir, val_split)

    train_manifest = merged_dir / "train.jsonl"
    val_manifest = merged_dir / "val.jsonl"
    hours = manifest_hours(train_manifest, val_manifest)

    desired_hours = os.environ.get("DESIRED_TRAIN_HOURS", "")
    if desired_hours:
        print(f"[VoxCPM Docker] Collected {hours:.2f}h of audio (target: {float(desired_hours):.2f}h)")
    else:
        print(f"[VoxCPM Docker] Collected {hours:.2f}h of audio")

    return {
        "train_manifest": str(train_manifest.resolve()),
        "val_manifest": str(val_manifest.resolve()) if val_manifest.exists() else "",
        "hours": hours,
    }


def resolve_finetune_mode(hours: float) -> str:
    """Honors FINETUNE_MODE ('lora' or 'full'), but downgrades 'full' to 'lora'
    when the dataset doesn't clear FULL_FINETUNE_MIN_HOURS."""
    requested_mode = os.environ.get("FINETUNE_MODE", "lora").strip().lower()
    min_hours = float(os.environ.get("FULL_FINETUNE_MIN_HOURS", "500"))

    if requested_mode == "full" and hours < min_hours:
        print(
            f"[VoxCPM Docker] Requested full finetuning but only {hours:.2f}h of data "
            f"is available (< {min_hours:.0f}h). Proceeding with LoRA finetuning instead."
        )
        return "lora"

    return requested_mode


def apply_finetune_mode(config: dict, mode: str):
    if mode == "full":
        config.pop("lora", None)
        return

    lora_cfg = config.get("lora") if isinstance(config.get("lora"), dict) else {}
    lora_defaults = {
        "enable_lm": True,
        "enable_dit": True,
        "enable_proj": True,
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
    }
    for key, val in lora_defaults.items():
        lora_cfg.setdefault(key, val)
    config["lora"] = lora_cfg


def main():
    default_config_path = os.environ.get(
        "CONFIG_PATH", "conf/voxcpm_v2/voxcpm_finetune_lora.yaml"
    )
    config_path = Path(default_config_path)

    config = {}
    if config_path.exists():
        print(f"[VoxCPM Docker] Loading base configuration from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"[VoxCPM Docker] Warning: Config path '{config_path}' not found. Starting with empty base config.")

    env_mapping = {
        "PRETRAINED_PATH": ["pretrained_path"],
        "TRAIN_MANIFEST": ["train_manifest"],
        "VAL_MANIFEST": ["val_manifest"],
        "SAMPLE_RATE": ["sample_rate"],
        "OUT_SAMPLE_RATE": ["out_sample_rate"],
        "BATCH_SIZE": ["batch_size"],
        "GRAD_ACCUM_STEPS": ["grad_accum_steps"],
        "NUM_WORKERS": ["num_workers"],
        "NUM_ITERS": ["num_iters"],
        "MAX_STEPS": ["max_steps"],
        "LOG_INTERVAL": ["log_interval"],
        "VALID_INTERVAL": ["valid_interval"],
        "SAVE_INTERVAL": ["save_interval"],
        "LEARNING_RATE": ["learning_rate"],
        "LR": ["learning_rate"],
        "WEIGHT_DECAY": ["weight_decay"],
        "WARMUP_STEPS": ["warmup_steps"],
        "MAX_BATCH_TOKENS": ["max_batch_tokens"],
        "MAX_GRAD_NORM": ["max_grad_norm"],
        "SAVE_PATH": ["save_path"],
        "TENSORBOARD": ["tensorboard"],
        "HF_MODEL_ID": ["hf_model_id"],
        "DISTRIBUTE": ["distribute"],
        "HUB_REPO_ID": ["hub_repo_id"],
        "LORA_ENABLE_LM": ["lora", "enable_lm"],
        "LORA_ENABLE_DIT": ["lora", "enable_dit"],
        "LORA_ENABLE_PROJ": ["lora", "enable_proj"],
        "LORA_R": ["lora", "r"],
        "LORA_RANK": ["lora", "r"],
        "LORA_ALPHA": ["lora", "alpha"],
        "LORA_DROPOUT": ["lora", "dropout"],
        "LAMBDA_LOSS_DIFF": ["lambdas", "loss/diff"],
        "LAMBDAS_LOSS_DIFF": ["lambdas", "loss/diff"],
        "LAMBDA_LOSS_STOP": ["lambdas", "loss/stop"],
        "LAMBDAS_LOSS_STOP": ["lambdas", "loss/stop"],
    }

    for env_key, keys in env_mapping.items():
        if env_key in os.environ and os.environ[env_key] != "":
            val = parse_env_value(os.environ[env_key])
            update_nested_dict(config, keys, val)
            print(f"[VoxCPM Docker] Env override: {' -> '.join(keys)} = {val}")

    for env_key, env_val in os.environ.items():
        if env_key.startswith("VOXCPM_") and env_val != "":
            key_part = env_key[len("VOXCPM_"):].lower()
            keys = key_part.split("__") if "__" in key_part else [key_part]
            val = parse_env_value(env_val)
            update_nested_dict(config, keys, val)
            print(f"[VoxCPM Docker] Generic env override ({env_key}): {' -> '.join(keys)} = {val}")

    if "EXTRA_CONFIG_JSON" in os.environ and os.environ["EXTRA_CONFIG_JSON"].strip():
        try:
            extra_dict = json.loads(os.environ["EXTRA_CONFIG_JSON"])

            def recursive_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        recursive_update(d[k], v)
                    else:
                        d[k] = v

            recursive_update(config, extra_dict)
            print(f"[VoxCPM Docker] Applied EXTRA_CONFIG_JSON overrides: {extra_dict}")
        except Exception as e:
            print(f"[VoxCPM Docker] Error parsing EXTRA_CONFIG_JSON: {e}")

    if os.environ.get("HF_TOKEN") and not config.get("hub_token"):
        config["hub_token"] = os.environ["HF_TOKEN"]

    pipeline_result = run_data_pipeline()
    if pipeline_result:
        config["train_manifest"] = pipeline_result["train_manifest"]
        if pipeline_result["val_manifest"]:
            config["val_manifest"] = pipeline_result["val_manifest"]
        dataset_hours = pipeline_result["hours"]
    else:
        dataset_hours = manifest_hours(
            Path(config["train_manifest"]) if config.get("train_manifest") else None,
            Path(config["val_manifest"]) if config.get("val_manifest") else None,
        )

    finetune_mode = resolve_finetune_mode(dataset_hours)
    apply_finetune_mode(config, finetune_mode)
    print(f"[VoxCPM Docker] Finetuning mode: {finetune_mode}  ({dataset_hours:.2f}h available)")

    runtime_config_path = Path("/tmp/voxcpm_runtime_config.yaml")
    with open(runtime_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n[VoxCPM Docker] Final generated runtime configuration ({runtime_config_path}):")
    print("-" * 50)
    print(yaml.dump(config, default_flow_style=False))
    print("-" * 50)

    nproc = os.environ.get("NPROC_PER_NODE", "1")
    use_torchrun = os.environ.get("USE_TORCHRUN", "false").lower() == "true" or int(nproc) > 1

    extra_args = sys.argv[1:]
    python_exec = sys.executable

    if use_torchrun:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        cmd = [
            python_exec, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            "scripts/train_voxcpm_finetune.py",
            "--config_path", str(runtime_config_path),
        ] + extra_args
    else:
        cmd = [
            python_exec, "scripts/train_voxcpm_finetune.py",
            "--config_path", str(runtime_config_path),
        ] + extra_args

    print(f"[VoxCPM Docker] Executing command: {' '.join(cmd)}\n")
    sys.stdout.flush()
    sys.stderr.flush()

    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()