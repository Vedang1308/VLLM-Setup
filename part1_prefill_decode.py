import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    device,
    peak_memory_gb,
    reset_peak_memory,
    seed_everything,
    synthetic_tokens,
    time_cuda_safe,
)


@torch.no_grad()
def benchmark_case(model, prompt_length, output_length, dev):
    input_ids = synthetic_tokens(
        batch_size=1,
        seq_len=prompt_length,
        vocab_size=model.config.vocab_size,
        dev=dev,
    )

    reset_peak_memory()

    # -----------------------------
    # PREFILL
    # -----------------------------
    def prefill():
        return model(input_ids=input_ids, use_cache=True)

    outputs, prefill_sec = time_cuda_safe(prefill)

    past = outputs.past_key_values
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    # -----------------------------
    # DECODE
    # -----------------------------
    def decode():
        nonlocal past, next_token

        for _ in range(output_length):
            outputs = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

    _, decode_sec = time_cuda_safe(decode)

    return {
        "prompt_length": prompt_length,
        "output_length": output_length,
        "prefill_ms": prefill_sec * 1000,
        "decode_ms": decode_sec * 1000,
        "total_ms": (prefill_sec + decode_sec) * 1000,
        "tpot_ms": decode_sec * 1000 / output_length,
        "peak_memory_gb": peak_memory_gb(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--out", default="part1_prefill_decode.csv")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions to average (Rule 6)")
    args = parser.parse_args()

    seed_everything(42)
    dev = device()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_val = torch.float16 if dev == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype_val,
        ).to(dev)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_val,
        ).to(dev)
    model.eval()

    # Warm-up (Requirement 3: Perform at least one warm-up inference)
    benchmark_case(model, 32, 4, dev)

    rows = []
    for n in [128, 512, 1024]:
        for m in [32, 128, 256]:
            print(f"Running prompt={n}, output={m} ({args.repeats} repeats)...")
            repeat_results = []
            for _ in range(args.repeats):
                repeat_results.append(benchmark_case(model, n, m, dev))

            # Compute mean metrics across repetitions (Part 0, Rule 6)
            avg_result = {
                "prompt_length": n,
                "output_length": m,
                "prefill_ms": sum(r["prefill_ms"] for r in repeat_results) / len(repeat_results),
                "decode_ms": sum(r["decode_ms"] for r in repeat_results) / len(repeat_results),
                "total_ms": sum(r["total_ms"] for r in repeat_results) / len(repeat_results),
                "tpot_ms": sum(r["tpot_ms"] for r in repeat_results) / len(repeat_results),
                "peak_memory_gb": max(r["peak_memory_gb"] for r in repeat_results),
            }
            rows.append(avg_result)

    df = pd.DataFrame(rows)
    print("\nPart 1 Benchmark Results:")
    print(df.to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
