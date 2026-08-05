"""Stable indirect LSD radix sort for PyTorch MPS using dynamic Metal shaders.

This module mirrors the repository's WebGPU baseline: eight 4-bit passes, with
one histogram dispatch, one scan dispatch, and one stable scatter dispatch per
pass. Tensors use torch.int32 as the storage dtype because uint32 support across
PyTorch MPS operations is less portable; Metal kernels reinterpret the buffers as
`uint*` and therefore preserve all 32 bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

KEY_BITS = 32
RADIX_BITS = 4
RADIX_SIZE = 1 << RADIX_BITS
PASSES = KEY_BITS // RADIX_BITS
WORKGROUP_SIZE = 256
ITEMS_PER_THREAD = 1
SUPPORTED_DTYPES = (torch.int32,)

METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;


kernel void radix_histogram(
    device const uint* keys [[buffer(0)]],
    device const uint* input_indices [[buffer(1)]],
    device uint* histograms [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& shift [[buffer(4)]],
    uint position [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup atomic_uint local_histogram[16];

    if (lid < 16u) {
        atomic_store_explicit(&local_histogram[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (position < count) {
        uint source_index = input_indices[position];
        uint key = keys[source_index];
        uint digit = (key >> shift) & 15u;
        atomic_fetch_add_explicit(&local_histogram[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < 16u) {
        histograms[group * 16u + lid] = atomic_load_explicit(&local_histogram[lid], memory_order_relaxed);
    }
}

kernel void radix_scan(
    device const uint* histograms [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    constant uint& group_count [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint bin_totals[16];

    if (lid < 16u) {
        uint total = 0u;
        for (uint group = 0u; group < group_count; ++group) {
            uint address = group * 16u + lid;
            offsets[address] = total;
            total += histograms[address];
        }
        bin_totals[lid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        uint base = 0u;
        for (uint digit = 0u; digit < 16u; ++digit) {
            for (uint group = 0u; group < group_count; ++group) {
                offsets[group * 16u + digit] += base;
            }
            base += bin_totals[digit];
        }
    }
}

kernel void radix_scatter(
    device const uint* keys [[buffer(0)]],
    device const uint* input_indices [[buffer(1)]],
    device uint* output_indices [[buffer(2)]],
    device const uint* offsets [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& shift [[buffer(5)]],
    uint position [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup uint tile_digits[256];

    bool valid = position < count;
    uint source_index = 0u;
    uint digit = 16u;
    if (valid) {
        source_index = input_indices[position];
        uint key = keys[source_index];
        digit = (key >> shift) & 15u;
    }
    tile_digits[lid] = digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        uint rank = 0u;
        for (uint previous = 0u; previous < lid; ++previous) {
            if (tile_digits[previous] == digit) {
                ++rank;
            }
        }
        uint offset_address = group * 16u + digit;
        uint output_position = offsets[offset_address] + rank;
        output_indices[output_position] = source_index;
    }
}
"""

_LIBRARY: Any | None = None


def _mps_available() -> bool:
    return hasattr(torch, "mps") and torch.backends.mps.is_available()


def _compile_library() -> Any:
    global _LIBRARY
    if _LIBRARY is None:
        if not _mps_available():
            raise RuntimeError("PyTorch MPS is not available on this system")
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("This PyTorch build does not provide torch.mps.compile_shader")
        _LIBRARY = torch.mps.compile_shader(METAL_SOURCE)
    return _LIBRARY


@dataclass(frozen=True)
class KernelProperties:
    max_threads_per_threadgroup: int | None
    thread_execution_width: int | None
    static_thread_group_memory_length: int | None


class MPSRadixSort:
    def __init__(self, capacity: int, output: torch.Tensor | None = None):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = int(capacity)
        group_count = max(1, (self.capacity + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE)
        self._scratch = torch.empty(self.capacity, device="mps", dtype=torch.int32)
        self._output = self._prepare_constructor_output(output)
        self._histograms = torch.empty(group_count * RADIX_SIZE, device="mps", dtype=torch.int32)
        self._offsets = torch.empty(group_count * RADIX_SIZE, device="mps", dtype=torch.int32)
        library = _compile_library()
        self._histogram = library.radix_histogram
        self._scan = library.radix_scan
        self._scatter = library.radix_scatter
        self.kernel_properties = {
            "radix_histogram": self._kernel_properties(self._histogram),
            "radix_scan": self._kernel_properties(self._scan),
            "radix_scatter": self._kernel_properties(self._scatter),
        }

    @staticmethod
    def _kernel_properties(kernel: Any) -> KernelProperties:
        return KernelProperties(
            getattr(kernel, "max_threads_per_threadgroup", None),
            getattr(kernel, "thread_execution_width", None),
            getattr(kernel, "static_thread_group_memory_length", None),
        )

    def sort(
        self,
        keys: torch.Tensor,
        indices: torch.Tensor,
        length: int | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sort ``indices`` by indirectly read ``keys`` into a caller-visible output buffer.

        If ``output`` is provided, the final pass writes into that tensor and the
        returned value is ``output[:length]``. Otherwise the sorter reuses the
        output tensor supplied to ``__init__`` or allocates one internal output
        tensor for the sorter lifetime. In all cases, repeated calls may
        overwrite the same output tensor.
        """
        count = indices.numel() if length is None else int(length)
        output_buffer = self._output if output is None else output
        self._validate_inputs(keys, indices, output_buffer, count)
        if count == 0:
            return output_buffer[:0]

        group_count = (count + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        dispatch_size = group_count * WORKGROUP_SIZE
        histograms = self._histograms[: group_count * RADIX_SIZE]
        offsets = self._offsets[: group_count * RADIX_SIZE]
        scratch = self._scratch[:count]
        output_slice = output_buffer[:count]

        input_buffer = indices[:count]
        for pass_index in range(PASSES):
            shift = pass_index * RADIX_BITS
            self._histogram(keys, input_buffer, histograms, count, shift,
                            threads=dispatch_size, group_size=WORKGROUP_SIZE,
                            arg_casts={3: "int32", 4: "int32"})
            self._scan(histograms, offsets, group_count,
                       threads=WORKGROUP_SIZE, group_size=WORKGROUP_SIZE,
                       arg_casts={2: "int32"})
            destination = scratch if pass_index % 2 == 0 else output_slice
            self._scatter(keys, input_buffer, destination, offsets, count, shift,
                          threads=dispatch_size, group_size=WORKGROUP_SIZE,
                          arg_casts={4: "int32", 5: "int32"})
            input_buffer = destination
        return output_slice

    def _prepare_constructor_output(self, output: torch.Tensor | None) -> torch.Tensor:
        if output is None:
            return torch.empty(self.capacity, device="mps", dtype=torch.int32)
        self._validate_output_tensor(output, 0)
        if output.numel() < self.capacity:
            raise ValueError("constructor output capacity is smaller than sorter capacity")
        return output

    def _validate_inputs(
        self, keys: torch.Tensor, indices: torch.Tensor, output: torch.Tensor, length: int
    ) -> None:
        if not _mps_available():
            raise RuntimeError("PyTorch MPS is not available on this system")
        for name, tensor in (("keys", keys), ("indices", indices)):
            if tensor.device.type != "mps":
                raise TypeError(f"{name} must be an MPS tensor")
            if tensor.dim() != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if tensor.dtype not in SUPPORTED_DTYPES:
                raise TypeError(f"{name} must use torch.int32 storage")
        self._validate_output_tensor(output, length)
        if output.data_ptr() == indices.data_ptr():
            raise ValueError("output must not alias the input indices tensor")
        if length < 0:
            raise ValueError("length must be non-negative")
        if length > self.capacity:
            raise ValueError("length exceeds sorter capacity")
        if length > indices.numel():
            raise ValueError("length exceeds indices.numel()")

    @staticmethod
    def _validate_output_tensor(output: torch.Tensor, length: int) -> None:
        if output.device.type != "mps":
            raise TypeError("output must be an MPS tensor")
        if output.dim() != 1:
            raise ValueError("output must be one-dimensional")
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")
        if output.dtype not in SUPPORTED_DTYPES:
            raise TypeError("output must use torch.int32 storage")
        if output.numel() < length:
            raise ValueError("output is shorter than length")


def radix_sort_mps(
    keys: torch.Tensor, indices: torch.Tensor, output: torch.Tensor | None = None
) -> torch.Tensor:
    return MPSRadixSort(indices.numel(), output=output).sort(keys, indices)
