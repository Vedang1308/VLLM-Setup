import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_part1(path):
    df = pd.read_csv(path)

    # 1. Prompt Length vs. Prefill Latency
    plt.figure(figsize=(7, 5))
    for output_length in sorted(df["output_length"].unique()):
        subset = df[df["output_length"] == output_length]
        plt.plot(
            subset["prompt_length"],
            subset["prefill_ms"],
            marker="o",
            label=f"Output={output_length}",
        )
    plt.xlabel("Prompt Length (tokens)")
    plt.ylabel("Prefill Latency (ms)")
    plt.title("Task 1.2: Prompt Length vs. Prefill Latency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("part1_prompt_vs_prefill.pdf", bbox_inches="tight")
    plt.savefig("part1_prompt_vs_prefill.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Per-output individual prefill plots (starter format)
    for output_length in sorted(df["output_length"].unique()):
        subset = df[df["output_length"] == output_length]
        plt.figure(figsize=(7, 5))
        plt.plot(subset["prompt_length"], subset["prefill_ms"], marker="o")
        plt.xlabel("Prompt Length")
        plt.ylabel("Prefill Latency (ms)")
        plt.title(f"Prompt Length vs Prefill Latency (Output={output_length})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(f"part1_prefill_output_{output_length}.pdf", bbox_inches="tight")
        plt.close()

    # 2. Output Length vs. Decode Latency
    plt.figure(figsize=(7, 5))
    for prompt_length in sorted(df["prompt_length"].unique()):
        subset = df[df["prompt_length"] == prompt_length]
        plt.plot(
            subset["output_length"],
            subset["decode_ms"],
            marker="s",
            label=f"Prompt={prompt_length}",
        )
    plt.xlabel("Output Length (tokens)")
    plt.ylabel("Decode Latency (ms)")
    plt.title("Task 1.2: Output Length vs. Decode Latency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("part1_output_vs_decode.pdf", bbox_inches="tight")
    plt.savefig("part1_output_vs_decode.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Output Length vs. TPOT (Time Per Output Token)
    plt.figure(figsize=(7, 5))
    for prompt_length in sorted(df["prompt_length"].unique()):
        subset = df[df["prompt_length"] == prompt_length]
        plt.plot(
            subset["output_length"],
            subset["tpot_ms"],
            marker="^",
            label=f"Prompt={prompt_length}",
        )
    plt.xlabel("Output Length (tokens)")
    plt.ylabel("TPOT (ms / token)")
    plt.title("Task 1.2: Output Length vs. TPOT")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("part1_output_vs_tpot.pdf", bbox_inches="tight")
    plt.savefig("part1_output_vs_tpot.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated Part 1 plots: part1_prompt_vs_prefill, part1_output_vs_decode, part1_output_vs_tpot")


def plot_part2(path):
    df = pd.read_csv(path)

    # 1. Output Length vs Generation Latency
    plt.figure(figsize=(7, 5))
    for cache_value in [True, False]:
        subset = df[df["use_cache"] == cache_value]
        plt.plot(
            subset["output_length"],
            subset["latency_sec"],
            marker="o",
            label=f"use_cache={cache_value}",
        )
    plt.xlabel("Output Length (tokens)")
    plt.ylabel("Generation Latency (s)")
    plt.title("Task 2.2: KV Cache vs. Generation Latency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("part2_kv_latency.pdf", bbox_inches="tight")
    plt.savefig("part2_kv_latency.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Output Length vs ms per output token
    if "ms_per_token" in df.columns:
        plt.figure(figsize=(7, 5))
        for cache_value in [True, False]:
            subset = df[df["use_cache"] == cache_value]
            plt.plot(
                subset["output_length"],
                subset["ms_per_token"],
                marker="s",
                label=f"use_cache={cache_value}",
            )
        plt.xlabel("Output Length (tokens)")
        plt.ylabel("Latency per Output Token (ms)")
        plt.title("Task 2.2: KV Cache vs. Latency per Output Token")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig("part2_kv_ms_per_token.pdf", bbox_inches="tight")
        plt.savefig("part2_kv_ms_per_token.png", bbox_inches="tight", dpi=300)
        plt.close()

    # 3. Output Length vs Peak GPU Memory
    if "peak_memory_gb" in df.columns:
        plt.figure(figsize=(7, 5))
        for cache_value in [True, False]:
            subset = df[df["use_cache"] == cache_value]
            plt.plot(
                subset["output_length"],
                subset["peak_memory_gb"],
                marker="^",
                label=f"use_cache={cache_value}",
            )
        plt.xlabel("Output Length (tokens)")
        plt.ylabel("Peak GPU Memory (GB)")
        plt.title("Task 2.2: KV Cache vs. Peak GPU Memory")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig("part2_kv_memory.pdf", bbox_inches="tight")
        plt.savefig("part2_kv_memory.png", bbox_inches="tight", dpi=300)
        plt.close()
    print("Generated Part 2 plots: part2_kv_latency, part2_kv_ms_per_token, part2_kv_memory")


def plot_part3(path):
    df = pd.read_csv(path)

    metrics = [
        ("tokens_per_sec", "Tokens / second", "batch_vs_throughput"),
        ("latency_sec", "Latency (s)", "batch_vs_latency"),
        ("peak_memory_gb", "Peak GPU Memory (GB)", "batch_vs_memory"),
    ]

    for column, ylabel, prefix in metrics:
        plt.figure(figsize=(7, 5))
        plt.plot(
            df["batch_size"],
            df[column],
            marker="o",
            linewidth=2,
            color="#1f77b4",
        )
        plt.xlabel("Batch Size")
        plt.ylabel(ylabel)
        plt.title(f"Task 3.2: Batch Size vs {ylabel}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(f"{prefix}.pdf", bbox_inches="tight")
        plt.savefig(f"{prefix}.png", bbox_inches="tight", dpi=300)
        plt.close()
    print("Generated Part 3 plots: batch_vs_throughput, batch_vs_latency, batch_vs_memory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part1")
    parser.add_argument("--part2")
    parser.add_argument("--part3")
    args = parser.parse_args()

    if args.part1:
        plot_part1(args.part1)

    if args.part2:
        plot_part2(args.part2)

    if args.part3:
        plot_part3(args.part3)


if __name__ == "__main__":
    main()
