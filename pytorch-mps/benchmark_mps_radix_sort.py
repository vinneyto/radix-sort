from __future__ import annotations

import random
import sys
import time
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch is not installed; MPS test/benchmark was not run.")
    raise SystemExit(0)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mps_radix_sort import MPSRadixSort  # noqa: E402


def ready() -> bool:
    return hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def to_i32(values: list[int]) -> torch.Tensor:
    return torch.tensor([v if v < 2**31 else v - 2**32 for v in values], dtype=torch.int32)


def unsigned(v: int) -> int:
    return v & 0xFFFF_FFFF


def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(1234 + n)
    keys = [rng.randrange(0, 1 << 20) for _ in range(n)]
    indices = list(range(n)); rng.shuffle(indices)
    return to_i32(keys), to_i32(indices)


def cpu_reference(keys: torch.Tensor, indices: torch.Tensor) -> tuple[list[int], float]:
    keys_list = [unsigned(int(v)) for v in keys.tolist()]
    indices_list = [unsigned(int(v)) for v in indices.tolist()]
    start = time.perf_counter()
    expected = sorted(indices_list, key=lambda index: keys_list[index])
    return expected, time.perf_counter() - start


def main() -> int:
    if not ready():
        print("PyTorch MPS with torch.mps.compile_shader is not available; benchmark was not run.")
        return 0
    for n in [10_000, 100_000, 1_000_000]:
        keys_cpu, indices_cpu = make_data(n)
        keys = keys_cpu.to("mps"); indices = indices_cpu.to("mps")
        output = torch.empty(n, device="mps", dtype=torch.int32)
        sorter = MPSRadixSort(n, output=output)
        for _ in range(3):
            sorter.sort(keys, indices)
        torch.mps.synchronize()
        repeats = 10 if n <= 100_000 else 5
        samples = []
        started = time.perf_counter()
        for _ in range(repeats):
            sample_start = time.perf_counter()
            result = sorter.sort(keys, indices)
            torch.mps.synchronize()
            samples.append(time.perf_counter() - sample_start)
        elapsed = time.perf_counter() - started
        expected, cpu_time = cpu_reference(keys_cpu, indices_cpu)
        actual = [unsigned(int(v)) for v in result.cpu().tolist()]
        ok = actual == expected
        avg = elapsed / repeats
        best = min(samples)
        print(f"n={n:,} repeats={repeats} avg_gpu={avg*1000:.3f} ms min_gpu={best*1000:.3f} ms cpu_ref={cpu_time*1000:.3f} ms speedup={cpu_time/avg:.2f}x verified={ok}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
