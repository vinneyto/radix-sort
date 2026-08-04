from __future__ import annotations

import random
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch is not installed; MPS test/benchmark was not run.")
    raise SystemExit(0)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mps_radix_sort import MPSRadixSort, radix_sort_mps  # noqa: E402

SIZES = [0, 1, 2, 15, 16, 17, 255, 256, 257, 511, 512, 513, 1_000, 10_000, 100_000, 1_000_000]


def mps_ready() -> bool:
    return hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def to_i32(values: list[int]) -> torch.Tensor:
    return torch.tensor([v if v < 2**31 else v - 2**32 for v in values], dtype=torch.int32)


def unsigned_value(v: int) -> int:
    return v & 0xFFFF_FFFF


def make_case(name: str, n: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    rng = random.Random(0xBAD5EED + n + hash(name) % 10000)
    length = n
    if name == "random32":
        keys = [rng.randrange(0, 2**32) for _ in range(max(n, 1))]
        indices = list(range(n))
        rng.shuffle(indices)
    elif name == "duplicates":
        keys = [rng.randrange(0, 32) for _ in range(max(n, 1))]
        indices = list(range(n)); rng.shuffle(indices)
    elif name == "all_equal":
        keys = [7 for _ in range(max(n, 1))]
        indices = list(range(n)); rng.shuffle(indices)
    elif name == "sorted":
        keys = list(range(max(n, 1)))
        indices = list(range(n))
    elif name == "reverse":
        keys = list(reversed(range(max(n, 1))))
        indices = list(range(n))
    elif name == "repeated_indices":
        key_count = max(1, min(n, 1024))
        keys = [rng.randrange(0, 256) for _ in range(key_count)]
        indices = [rng.randrange(0, key_count) for _ in range(n)]
    elif name == "partial_length":
        keys = [rng.randrange(0, 128) for _ in range(max(n, 1))]
        indices = list(range(n)); rng.shuffle(indices)
        length = max(0, n - 3)
    else:
        raise ValueError(name)
    return to_i32(keys), to_i32(indices), length


def assert_case(case_name: str, n: int, keys_cpu: torch.Tensor, indices_cpu: torch.Tensor, length: int) -> None:
    keys_mps = keys_cpu.to("mps")
    indices_mps = indices_cpu.to("mps")
    sorter = MPSRadixSort(indices_cpu.numel())
    result = sorter.sort(keys_mps, indices_mps, length=length)
    torch.mps.synchronize()
    actual = [unsigned_value(int(v)) for v in result.cpu().tolist()]
    keys_list = [unsigned_value(int(v)) for v in keys_cpu.tolist()]
    input_indices = [unsigned_value(int(v)) for v in indices_cpu[:length].tolist()]
    expected = sorted(input_indices, key=lambda index: keys_list[index])
    if actual != expected:
        for pos, (e, a) in enumerate(zip(expected, actual)):
            if e != a:
                print(f"FAILED {case_name} size={n}")
                print(f"first mismatch: {pos}")
                print(f"expected index: {e}")
                print(f"actual index: {a}")
                print(f"expected key: {keys_list[e]}")
                print(f"actual key: {keys_list[a]}")
                raise AssertionError("radix sort output mismatch")
        raise AssertionError("radix sort output length mismatch")


def main() -> int:
    if not mps_ready():
        print("PyTorch MPS with torch.mps.compile_shader is not available; GPU tests were not run.")
        return 0
    smoke_keys = to_i32([3, 1, 2]).to("mps")
    smoke_indices = to_i32([0, 1, 2]).to("mps")
    assert [int(v) for v in radix_sort_mps(smoke_keys, smoke_indices).cpu().tolist()] == [1, 2, 0]
    passed = 1
    for n in SIZES:
        for case_name in ["random32", "duplicates", "all_equal", "sorted", "reverse", "repeated_indices", "partial_length"]:
            assert_case(case_name, n, *make_case(case_name, n))
            passed += 1
    print(f"all tests passed; cases: {passed}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
