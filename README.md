# CSE 598: Data Intensive Systems for Machine Learning — LLM Serving Systems Assignment

This repository contains the complete implementation, benchmarks, empirical results, and system design artifacts for **Coding Assignment 3: LLM Serving**.

All experiments were executed on **ASU's SOL Supercomputer** utilizing an **NVIDIA A100-SXM4-80GB GPU**.

---

## 1. System & Environment Specifications

- **Compute Node**: ASU SOL HPC Cluster (Interactive GPU Partition)
- **Accelerator**: NVIDIA A100-SXM4-80GB (79.28 GiB VRAM, Compute Capability 8.0)
- **Host OS**: Linux (RHEL 8.10 x86_64, Kernel 4.18.0)
- **Python**: 3.13.11 (Miniconda)
- **PyTorch**: 2.9.1 (CUDA 12.8)
- **Triton**: 3.5.1
- **Hugging Face Transformers**: 5.12.1
- **vLLM**: 0.15.1
- **Accelerate**: 1.14.0

---

## 2. Repository File Structure

```text
.
├── requirements.txt              # Python package dependencies
├── utils.py                      # Common timing, synchronization, and memory utilities
├── workload.py                   # Poisson arrival workload generator
├── workload.json                 # Generated 50-request heterogeneous workload
│
├── part0_env_check.py            # Part 0: Environment verification & Question 0.1 benchmark
│
├── part1_prefill_decode.py       # Part 1: Prefill & Decode benchmark (3-repeat mean)
├── part1_prefill_decode.csv      # Part 1: Empirical results across prompt & output lengths
├── part1_prompt_vs_prefill.pdf   # Task 1.2: Prompt Length vs. Prefill Latency
├── part1_output_vs_decode.pdf    # Task 1.2: Output Length vs. Decode Latency
├── part1_output_vs_tpot.pdf      # Task 1.2: Output Length vs. TPOT
│
├── part2_kv_cache.py             # Part 2: KV Cache implementation & agreement check
├── part2_kv_cache.csv            # Part 2: Cached vs. Uncached benchmark measurements
├── part2_kv_size.py              # Part 2: KV Cache analytical memory sizing tool
├── part2_kv_latency.pdf          # Task 2.2: Generation Latency vs. Output Length
├── part2_kv_ms_per_token.pdf     # Task 2.2: Latency per Output Token vs. Output Length
├── part2_kv_memory.pdf           # Task 2.2: Peak GPU Memory vs. Output Length
│
├── part3_batching.py             # Part 3: Static Batching benchmark (batch sizes 1 to 16)
├── part3_batching.csv            # Part 3: Empirical results (throughput, latency, memory)
├── batch_vs_throughput.pdf       # Task 3.2: Batch Size vs. Tokens / second
├── batch_vs_latency.pdf          # Task 3.2: Batch Size vs. End-to-End Latency
├── batch_vs_memory.pdf           # Task 3.2: Batch Size vs. Peak GPU Memory
│
├── part4_static_batching.py      # Part 4: Static batching runner on heterogeneous workload
├── part4_vllm_offline.py         # Part 4: Modern engine (vLLM) offline batch benchmark
│
├── report.pdf                    # Deliverable 2: Complete submission PDF report (15 pages)
├── report.html                   # HTML source for report
├── plot_results.py               # Central plotting engine for Parts 1, 2, and 3
└── README.md                     # Experiment reproduction and documentation guide
```

---

## 3. How to Reproduce All Experiments on ASU SOL

### Prerequisites & Node Allocation
Request an interactive GPU session with an NVIDIA A100 on SOL:
```bash
# Example Slurm interactive allocation
srun -p gpu -N 1 --gpus=1 --time=02:00:00 --pty bash
```

Clone the repository and set temporary directories:
```bash
git clone https://github.com/Vedang1308/VLLM-Setup.git
cd VLLM-Setup

# Use scratch storage to prevent home quota exhaustion
export TMPDIR=/scratch/$USER/tmp
mkdir -p $TMPDIR
```

Install core dependencies:
```bash
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 triton==3.5.1
pip install --no-cache-dir "numpy==2.2.6"
pip install --no-cache-dir vllm
```

---

### Step-by-Step Execution Guide

#### Part 0: Environment Verification & Question 0.1
```bash
python part0_env_check.py
```
- Logs GPU device name, compute capability, VRAM, and installed library versions.
- Demonstrates the empirical latency difference between asynchronous CPU launch (`~9.0 ms`) vs. synchronized GPU completion (`~6.8 ms`).

#### Part 1: Prefill and Decode Scaling
```bash
# Run benchmark across prompt lengths (128, 512, 768) and output lengths (32, 128, 256)
python part1_prefill_decode.py --repeats 3

# Generate Task 1.2 plots (saves both PDF and PNG)
python plot_results.py --part1 part1_prefill_decode.csv
```

#### Part 2: KV Cache Evaluation & Sizing
```bash
# Run cached vs. uncached benchmark and automated token agreement check
python part2_kv_cache.py --model gpt2 --repeats 3

# Generate Task 2.2 plots
python plot_results.py --part2 part2_kv_cache.csv

# Verify Task 2.3 memory formula
python part2_kv_size.py --batch-size 1 --layers 32 --seq-len 4096 --kv-heads 8 --head-dim 128 --bytes 2
```

#### Part 3: Static Batching Scalability
```bash
# Evaluate batch sizes [1, 2, 4, 8, 16] with 3-repetition averaging
python part3_batching.py --prompt-length 128 --output-length 128 --repeats 3

# Generate Task 3.2 plots
python plot_results.py --part3 part3_batching.csv
```

#### Part 4: Heterogeneous Requests & Serving Engines
```bash
# 1. Generate the 50-request Poisson arrival heterogeneous workload
python workload.py --n 50 --seed 42

# 2. Run naive static batching (batch size 4)
python part4_static_batching.py --batch-size 4

# 3. Run modern serving engine (vLLM offline batch)
python part4_vllm_offline.py
```

---

## 4. Summary of Empirical Results

| Part & Experiment | Key Metrics (NVIDIA A100-SXM4-80GB) | Primary Systems Insight |
| :--- | :--- | :--- |
| **Part 1: Prefill vs. Decode** | Prefill: $7.6 - 11.6\text{ ms}$; TPOT: $6.28 - 6.35\text{ ms/tok}$ | Prefill is parallel & compute-bound; Decode is sequential & memory-bandwidth bound. |
| **Part 2: KV Cache** | 100% Token Agreement; Peak Memory: $0.263\text{ GB}$ (cached) vs $0.319\text{ GB}$ (uncached) | Uncached decoding recomputes entire prefix, multiplying memory activations across layers. |
| **Part 3: Batching** | Throughput: $159 \to 2,641\text{ tok/s}$ ($16.6\times$ gain); Latency: flat at $\sim 0.77\text{ s}$ | Batching amortizes memory loads of model weights across tokens, drastically lifting throughput. |
| **Part 4: Static vs. vLLM** | Static: $4.74\text{ req/s}$, $50.2\%$ pad efficiency; vLLM: $\mathbf{56.44\text{ req/s}}$ ($\mathbf{11.9\times}$ speedup) | Continuous batching & PagedAttention eliminate padding bubbles and memory fragmentation. |

---

## 5. Written Report & Deliverables
- **Official Submission PDF Report**: [`report.pdf`](report.pdf) *(15-page publication-grade report containing all methodology, Task 1.3/2.4/3.3 analyses, Task 2.3 derivation, Task 4 comparison table, Part 5 serving policy design, Part 6 reflection, and embedded plots)*
- **HTML Report Source**: [`report.html`](report.html)
