# PyTorch MPS Metal radix sort

This directory is a standalone PyTorch/MPS example of the repository's current
WebGPU baseline. It implements a stable indirect LSD radix sort over 32-bit keys
and compiles the Metal kernels directly from Python with
`torch.mps.compile_shader(...)`; it does not use C++, Objective-C++, `setup.py`,
or `torch.utils.cpp_extension`.

## Requirements

- macOS on Apple Silicon or another Mac supported by PyTorch's MPS backend.
- A PyTorch build containing `torch.mps.compile_shader`.
- An available MPS device, which you can check with:

```sh
python - <<'PY'
import torch
print(torch.backends.mps.is_available())
print(hasattr(torch.mps, "compile_shader"))
PY
```

Install the only Python dependency with:

```sh
python -m pip install -r pytorch-mps/requirements.txt
```

## Algorithm and tensor contract

`radix_sort_mps(keys, indices, output=None)` and
`MPSRadixSort.sort(..., output=None)` sort only
`indices`; `keys` remains immutable and is read indirectly as
`keys[indices[position]]`. The result is stable, so equal keys preserve their
relative order from the input index sequence.

For WebGPU-like caller ownership, create the output tensor outside the sorter and
pass it either to `MPSRadixSort(capacity, output=output)` or to an individual
`sort(..., output=output)` call. The final radix pass writes into that tensor and
the returned value is `output[:length]`. If no output tensor is supplied, the
sorter allocates one internal output tensor once and reuses it; `sort()` does not
allocate a fresh result buffer for every call.

The baseline configuration matches `src/RadixSort.ts`:

- `KEY_BITS = 32`
- `RADIX_BITS = 4`
- `RADIX_SIZE = 16`
- `PASSES = 8`
- `WORKGROUP_SIZE = 256`
- `ITEMS_PER_THREAD = 1`

The Python tensors use `torch.int32` as the storage dtype. The Metal kernels bind
those buffers as `device const uint*` / `device uint*`, so all 32 bits are treated
as unsigned key and index data inside Metal. This avoids relying on PyTorch MPS
`torch.uint32` operation coverage outside the custom kernels.

## Metal kernels

Each radix pass is three ordered dispatches:

1. `radix_histogram` builds one 16-bin threadgroup-local atomic histogram for
   each 256-element tile.
2. `radix_scan` runs as one 256-thread threadgroup, computes an exclusive prefix
   per `(group, digit)`, and adds each digit's global base.
3. `radix_scatter` computes a stable local rank by counting earlier lanes in the
   same tile with the same digit, then writes the source index to the scanned
   output position.

Ping-pong buffers match the WebGPU example: pass 0 writes original to scratch,
pass 1 writes scratch to the caller-visible output tensor, and the pattern repeats
until pass 7 leaves the final result in the output tensor returned by `sort()`.

## Run tests and benchmark

From the repository root:

```sh
python pytorch-mps/test_mps_radix_sort.py
python pytorch-mps/benchmark_mps_radix_sort.py
```

Or from this directory:

```sh
python test_mps_radix_sort.py
python benchmark_mps_radix_sort.py
```

If MPS or `torch.mps.compile_shader` is unavailable, the scripts print a clear
message and exit without reporting a false success.

## Current limitations

This is intentionally a correctness-first baseline. It does not implement
SIMD-group stable rank, multiple items per thread, multiple tiles per
threadgroup, processed-earlier-tile accumulation, hierarchical scan, 8-bit radix,
64-bit keys, pass skipping, or autograd/backward support.
