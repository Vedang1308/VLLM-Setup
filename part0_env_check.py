"""
Part 0: Environment and Methodology Verification Script

This script records all system and library specifications required by Part 0:
1. GPU model and device properties
2. CUDA, PyTorch, Transformers, and vLLM versions
3. Performs warm-up inference
4. Empirically demonstrates Question 0.1: asynchronous GPU kernel execution
   vs synchronized execution timing.
"""

import sys
# Prevent broken torchaudio shared library mismatch from crashing transformers
sys.modules.setdefault("torchaudio", None)

import platform
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

def record_environment():
    lines = []
    lines.append("=" * 60)
    lines.append("PART 0: ENVIRONMENT & METHODOLOGY SPECIFICATION")
    lines.append("=" * 60)

    # Python & OS
    py_ver = sys.version.replace("\n", " ")
    lines.append(f"Python Version:       {py_ver}")
    lines.append(f"OS / Platform:        {platform.platform()}")

    # PyTorch & CUDA
    if TORCH_AVAILABLE:
        lines.append(f"PyTorch Version:      {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        lines.append(f"CUDA Available:       {cuda_avail}")
        if cuda_avail:
            lines.append(f"CUDA Version (PyTorch): {torch.version.cuda}")
            lines.append(f"CUDA Device Count:    {torch.cuda.device_count()}")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_mem_gb = gpu_mem_bytes / (1024 ** 3)
            compute_cap = torch.cuda.get_device_capability(0)
            lines.append(f"GPU Model:            {gpu_name}")
            lines.append(f"GPU Total Memory:     {gpu_mem_gb:.2f} GB ({gpu_mem_bytes} bytes)")
            lines.append(f"GPU Compute Cap:      {compute_cap[0]}.{compute_cap[1]}")
        else:
            lines.append("WARNING: No CUDA GPU detected! Ensure you are running on a GPU node.")
    else:
        lines.append("PyTorch Version:      NOT INSTALLED (pip install torch)")
        lines.append("CUDA Available:       False (torch missing)")

    # Transformers
    try:
        import transformers
        lines.append(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        lines.append("Transformers Version: NOT INSTALLED")

    # vLLM
    try:
        import vllm
        lines.append(f"vLLM Version:         {vllm.__version__}")
    except ImportError:
        lines.append("vLLM Version:         NOT INSTALLED (install with: pip install vllm)")

    # Accelerate
    try:
        import accelerate
        lines.append(f"Accelerate Version:   {accelerate.__version__}")
    except ImportError:
        pass

    lines.append("=" * 60)
    summary_text = "\n".join(lines)
    print(summary_text)


def demo_question_0_1():
    """
    Empirical demonstration of Question 0.1:
    Demonstrates why timing without torch.cuda.synchronize() measures
    CPU kernel launch latency rather than actual GPU execution time.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("\nSkipping Question 0.1 demo: PyTorch or CUDA not available on this machine.")
        return

    from transformers import AutoModelForCausalLM

    print("\nRunning Question 0.1 demonstration (GPT-2 on CUDA)...")
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    input_ids = torch.randint(0, model.config.vocab_size, (1, 512), device=device)

    # Warm-up (Requirement 3: Perform at least one warm-up inference)
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize()

    # Timing WITHOUT synchronize (Incorrect timing)
    torch.cuda.synchronize()
    start_unsync = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids)
    end_unsync = time.perf_counter()
    elapsed_unsync_ms = (end_unsync - start_unsync) * 1000

    # Timing WITH synchronize (Correct timing)
    torch.cuda.synchronize()
    start_sync = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize()
    end_sync = time.perf_counter()
    elapsed_sync_ms = (end_sync - start_sync) * 1000

    demo_result = (
        f"\n--- Question 0.1 Empirical Measurement ---\n"
        f"Input shape: [batch=1, seq_len=512], Model: gpt2 (fp16)\n"
        f"Time WITHOUT torch.cuda.synchronize() : {elapsed_unsync_ms:8.4f} ms  (CPU launch / enqueuing time only)\n"
        f"Time WITH    torch.cuda.synchronize() : {elapsed_sync_ms:8.4f} ms  (Actual GPU compute completion time)\n"
        f"Ratio (Sync / Unsync): {elapsed_sync_ms / max(elapsed_unsync_ms, 1e-6):.1f}x difference!\n"
    )
    print(demo_result)


if __name__ == "__main__":
    record_environment()
    demo_question_0_1()
