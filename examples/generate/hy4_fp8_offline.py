# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline inference example for Hy4-preview-FP8 (tencent/Hy4-preview-FP8) on XPU.

Usage (8 GPUs, recommended):
    python examples/generate/hy4_offline.py \
        --model tencent/Hy4-preview-FP8 \
        --tensor-parallel-size 8

With MTP speculative decoding:
    python examples/generate/hy4_offline.py \
        --model tencent/Hy4-preview-FP8 \
        --tensor-parallel-size 8 \
        --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'

All engine arguments (e.g. --max-model-len, --gpu-memory-utilization) are
accepted via the CLI.
"""

import json
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser(
        description="Hy4-preview-FP8 offline inference example")
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="tencent/Hy4-preview-FP8",
        trust_remote_code=True,
        dtype="auto",
        seed=1234,
    )

    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=256)
    sampling_group.add_argument("--temperature", type=float, default=0.9)
    sampling_group.add_argument("--top-p", type=float, default=1.0)
    sampling_group.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic decoding (temperature=0, top_p=1).",
    )
    return parser


def main(args: dict):
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    greedy = args.pop("greedy")
    if greedy:
        temperature = 0.0
        top_p = 1.0

    llm = LLM(**args)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # --- Plain text completion ---
    prompts = [
        "Explain quantum entanglement in simple terms.",
        "Write a short poem about the ocean.",
        "What are the main differences between Python and Rust?",
        "Summarize the theory of general relativity in one paragraph.",
    ]

    print("=" * 80)
    print("Plain text completion (Hy4-preview-FP8)")
    print("=" * 80)
    print(
        f"Config: seed={args.get('seed')}, max_tokens={max_tokens}, "
        f"temperature={temperature}, top_p={top_p}, greedy={greedy}"
    )
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"\nPrompt:    {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 80)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)

