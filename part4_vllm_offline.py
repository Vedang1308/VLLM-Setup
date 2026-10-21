"""
This file demonstrates a minimal vLLM comparison.

Install separately:
    pip install vllm

Note:
Offline llm.generate() is convenient for throughput experiments, but a true
request-arrival / TTFT experiment is better performed with the vLLM server
and an asynchronous client. For this assignment, students may use offline
generation unless the instructor requires online arrival-time measurements.
"""

import sys
# Prevent ABI mismatches in unused audio/vision libraries from crashing text LLM loading
sys.modules.setdefault("torchaudio", None)
sys.modules.setdefault("torchvision", None)

import argparse
import json
import time

from vllm import LLM, SamplingParams


def load_workload(path):
    with open(path) as f:
        return json.load(f)


def make_prompt(n_words):
    # This approximates a prompt of the desired scale but is not guaranteed
    # to tokenize to exactly n_words tokens.
    return "hello " * n_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--workload", default="workload.json")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "auto"], help="Numerical precision")
    args = parser.parse_args()

    workload = load_workload(args.workload)

    # Ensure prompt + max_tokens fits within model context limit (1024 for GPT-2)
    max_context = 1024
    safe_prompt_cap = max(16, max_context - args.max_tokens - 4)

    prompts = [
        make_prompt(min(r["prompt_length"], safe_prompt_cap))
        for r in workload
    ]

    llm = LLM(model=args.model, dtype=args.dtype)

    # Simple common max-token setting for an offline comparison.
    # TODO 4:
    # Explain why this does not perfectly reproduce heterogeneous output lengths.
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
    )

    start = time.perf_counter()

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    end = time.perf_counter()

    total_time = end - start

    output_tokens = sum(
        len(output.outputs[0].token_ids)
        for output in outputs
    )

    print(f"Requests/s: {len(prompts) / total_time:.3f}")
    print(f"Output tokens/s: {output_tokens / total_time:.3f}")
    print(f"Total time: {total_time:.3f} s")


if __name__ == "__main__":
    main()
