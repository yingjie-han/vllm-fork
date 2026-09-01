# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DEFAULT_MODEL = "tencent/Hy4-preview"
DEFAULT_NUM_LAYERS = 3
DEFAULT_NUM_EXPERTS = 8
DEFAULT_FIXTURE_DIR = Path.home() / ".cache" / "vllm" / "hy4-xpu-reduced"

TEXT_ASSET_PATTERNS = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
WEIGHT_FILE_PATTERNS = (
    "*.bin",
    "*.bin.index.json",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.safetensors.index.json",
)
PER_LAYER_FIELDS = ("layer_types", "mlp_layer_types", "indexer_types")
MUTABLE_FIELDS = {
    "num_hidden_layers",
    "num_nextn_predict_layers",
    "n_routed_experts",
    "num_experts_per_tok",
    *PER_LAYER_FIELDS,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reduced three-layer Hy4 model with random BF16 weights "
            "on one Intel XPU."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="Directory for non-weight model assets and the reduced config.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        choices=(2, 3),
        default=DEFAULT_NUM_LAYERS,
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=DEFAULT_NUM_EXPERTS,
        help="Reduced routed-expert count; 8 is sized for a single 24 GiB XPU.",
    )
    parser.add_argument("--prompt", default="Explain why the sky appears blue.")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def _assert_no_weight_files(root: Path) -> None:
    weight_files = sorted(
        path
        for pattern in WEIGHT_FILE_PATTERNS
        for path in root.rglob(pattern)
        if path.is_file()
    )
    if weight_files:
        paths = ", ".join(str(path.relative_to(root)) for path in weight_files)
        raise RuntimeError(f"Weight files are not allowed in the fixture: {paths}")


def _copy_text_assets(snapshot_dir: Path, fixture_dir: Path) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for filename in TEXT_ASSET_PATTERNS:
        source = snapshot_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required Hy4 text asset is missing: {filename}")
        shutil.copy2(source, fixture_dir / filename)


def _validate_reduced_config(
    original: dict[str, Any],
    reduced: dict[str, Any],
    num_layers: int,
    num_experts: int,
) -> None:
    for field, value in original.items():
        if field not in MUTABLE_FIELDS and reduced.get(field) != value:
            raise ValueError(f"Reduced fixture changed production field {field!r}")

    for field in PER_LAYER_FIELDS:
        if reduced[field] != original[field][:num_layers]:
            raise ValueError(f"Reduced fixture did not correctly truncate {field!r}")

    if reduced["n_routed_experts"] != num_experts:
        raise ValueError("Reduced fixture has the wrong routed-expert count")


def build_reduced_config(
    config: dict[str, Any],
    num_layers: int,
    num_experts: int,
) -> dict[str, Any]:
    if num_layers not in (2, 3):
        raise ValueError("Hy4 XPU fixture supports two or three layers")
    if num_experts < config["num_experts_per_tok"]:
        raise ValueError(
            "num_experts must be at least num_experts_per_tok "
            f"({config['num_experts_per_tok']})"
        )
    if num_experts > config["n_routed_experts"]:
        raise ValueError(
            f"num_experts cannot exceed {config['n_routed_experts']}"
        )

    reduced = json.loads(json.dumps(config))
    reduced["num_hidden_layers"] = num_layers
    reduced["num_nextn_predict_layers"] = 0
    reduced["n_routed_experts"] = num_experts
    for field in PER_LAYER_FIELDS:
        reduced[field] = reduced[field][:num_layers]
    reduced.pop("quantization_config", None)

    _validate_reduced_config(config, reduced, num_layers, num_experts)
    return reduced


def materialize_reduced_model(
    model: str,
    fixture_dir: Path,
    num_layers: int,
    num_experts: int,
    revision: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from vllm.transformers_utils.repo_utils import hf_api

    snapshot_dir = Path(
        hf_api().snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=list(TEXT_ASSET_PATTERNS),
            ignore_patterns=list(WEIGHT_FILE_PATTERNS),
        )
    )

    if fixture_dir.exists():
        _assert_no_weight_files(fixture_dir)
    _copy_text_assets(snapshot_dir, fixture_dir)

    config_path = fixture_dir / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    reduced_config = build_reduced_config(config, num_layers, num_experts)
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(reduced_config, config_file, indent=2)
        config_file.write("\n")

    _assert_no_weight_files(fixture_dir)
    return fixture_dir, config, reduced_config


def estimate_bf16_weight_gib(config: dict[str, Any]) -> float:
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    expert_size = config["moe_intermediate_size"]
    num_experts = config["n_routed_experts"]
    vocab_size = config["vocab_size"]

    parameter_count = 2 * vocab_size * hidden_size
    for mlp_type in config["mlp_layer_types"]:
        if mlp_type == "dense":
            parameter_count += 3 * hidden_size * intermediate_size
        else:
            parameter_count += 3 * hidden_size * expert_size * (num_experts + 1)
            parameter_count += hidden_size * num_experts
    return parameter_count * 2 / 1024**3


def main() -> None:
    import torch

    from vllm import LLM, SamplingParams

    args = _parse_args()
    model_path, original_config, reduced_config = materialize_reduced_model(
        args.model,
        args.fixture_dir,
        args.num_layers,
        args.num_experts,
        args.revision,
    )
    print("Reduced Hy4 shape:")
    print(
        "  num_hidden_layers: "
        f"{original_config['num_hidden_layers']} -> "
        f"{reduced_config['num_hidden_layers']}"
    )
    print(
        "  n_routed_experts: "
        f"{original_config['n_routed_experts']} -> "
        f"{reduced_config['n_routed_experts']}"
    )
    print(
        "  num_experts_per_tok: "
        f"{reduced_config['num_experts_per_tok']} (unchanged)"
    )
    print(
        "  num_nextn_predict_layers: "
        f"{original_config['num_nextn_predict_layers']} -> "
        f"{reduced_config['num_nextn_predict_layers']}"
    )
    for field in PER_LAYER_FIELDS:
        print(
            f"  {field}: {len(original_config[field])} entries -> "
            f"{reduced_config[field]}"
        )
    print(
        "Estimated embedding/head/MLP BF16 weights: "
        f"{estimate_bf16_weight_gib(reduced_config):.2f} GiB"
    )

    free_memory_before, _ = torch.xpu.mem_get_info()
    init_start = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        load_format="dummy",
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    torch.xpu.synchronize()
    init_seconds = time.perf_counter() - init_start
    free_memory_after_init, _ = torch.xpu.mem_get_info()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    generation_start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params, use_tqdm=False)
    torch.xpu.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    free_memory_after_generation, _ = torch.xpu.mem_get_info()

    result = outputs[0].outputs[0]
    observed_memory_gib = (
        free_memory_before
        - min(free_memory_after_init, free_memory_after_generation)
    ) / 1024**3
    print(f"Token IDs: {result.token_ids}")
    print(f"Generated text: {result.text!r}")
    print(f"Initialization: {init_seconds:.3f} s")
    print(f"One-request generation: {generation_seconds:.3f} s")
    print(f"Observed XPU memory increase: {observed_memory_gib:.2f} GiB")


if __name__ == "__main__":
    main()