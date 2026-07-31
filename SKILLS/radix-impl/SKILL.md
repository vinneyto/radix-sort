---
name: gpu-radix-sort
description: Implement a portable, stable indirect LSD radix sort over an abstract GPU compute API such as WebGPU, Metal, Vulkan, CUDA, HIP, or DirectCompute. Use when sorting an index buffer according to keys read indirectly from a separate immutable key buffer.
---

# GPU Radix Sort Skill

Implement a stable least-significant-digit indirect radix sort using only common GPU compute primitives.

The primary contract is to reorder an index buffer according to keys stored in a separate immutable key buffer. At every sorting position, the current key is obtained through indirect lookup:

```text
sourceIndex = inputIndices[position]
key = keyBuffer[sourceIndex]
```

Only indices are scattered between ping-pong buffers. The key buffer is never reordered or copied by the sort.

Keep the sorting algorithm independent from the concrete API. API-specific code should be isolated behind a thin adapter responsible for resource creation, bindings, pipeline creation, dispatch, and inter-dispatch synchronization.

## Supported targets

The algorithm should be portable to compute APIs that provide:

- compute kernels;
- workgroups or threadgroups;
- workgroup-local shared memory;
- barriers within a workgroup;
- read/write storage buffers;
- atomic integer operations;
- multiple ordered dispatches;
- visibility of storage writes to later dispatches.

Subgroup operations such as ballot, shuffle, prefix sum, or SIMD-group scans are optional optimizations. They must not be required for correctness.

## Use this skill when

Use this skill to:

- implement a GPU radix sort from scratch;
- port an existing Vulkan, CUDA, Metal, or WebGPU radix sort;
- separate the algorithm from API-specific command encoding;
- verify the stability and correctness of a GPU radix sort;
- optimize a correct baseline implementation;
- sort an index buffer by keys stored in a separate buffer using multiple workgroups.

Do not use this skill for comparison sorting, in-place CPU sorting, direct reordering of the key buffer, or a single-workgroup-only toy implementation unless that is explicitly the requested scope.

# 1. Define the indirect sorting contract

Start by defining the immutable key source, the index permutation, and the ordering semantics.

Required input:

```text
keyBuffer[M]
inputIndices[N]
```

Each entry in `inputIndices` identifies an element in `keyBuffer`:

```text
sourceIndex = inputIndices[position]
key = keyBuffer[sourceIndex]
```

The sort must reorder only the indices. It must not reorder or duplicate `keyBuffer`.

Required output:

```text
sortedIndices[N]
```

The output must satisfy:

```text
keyBuffer[sortedIndices[i]]
    <= keyBuffer[sortedIndices[i + 1]]
```

The result is therefore a permutation of the input index sequence, ordered by the indirectly referenced keys.

Example:

```text
keyBuffer:    [40, 10, 30]
inputIndices: [2, 0, 1]
```

The referenced key sequence is:

```text
[keyBuffer[2], keyBuffer[0], keyBuffer[1]]
= [30, 40, 10]
```

The sorted index result is:

```text
sortedIndices: [1, 2, 0]
```

because it represents the ordered key sequence `[10, 30, 40]` without moving `keyBuffer`.

The sort must also be stable:

```text
If two referenced keys compare equal, their relative order in inputIndices
must be preserved in sortedIndices.
```

Stability is mandatory for an LSD radix sort because every new pass must preserve the ordering produced by all previous, less-significant passes.

The index buffer does not have to begin as `[0, 1, 2, ...]`. It may contain any valid subset or permutation of indices into `keyBuffer`, including repeated indices if the caller intentionally permits them.

Every consumed index must satisfy `0 <= inputIndices[position] < M`. Out-of-range indices are an input-contract violation and must not be silently read.

# 2. Choose the radix configuration

Define:

```text
KEY_BITS
RADIX_BITS
NUM_BINS = 1 << RADIX_BITS
WORKGROUP_SIZE
ITEMS_PER_THREAD
ITEMS_PER_GROUP = WORKGROUP_SIZE * ITEMS_PER_THREAD
NUM_GROUPS = ceil(N / ITEMS_PER_GROUP)
NUM_PASSES = ceil(KEY_BITS / RADIX_BITS)
```

A common baseline is:

```text
KEY_BITS        = 32
RADIX_BITS      = 8
NUM_BINS        = 256
WORKGROUP_SIZE  = 256
ITEMS_PER_THREAD = 4 or 8
NUM_PASSES      = 4
```

Each pass processes one radix digit:

```text
shift = passIndex * RADIX_BITS
digit = (key >> shift) & passMask
```

For a full-width pass:

```text
passMask = NUM_BINS - 1
```

If `KEY_BITS` is not divisible by `RADIX_BITS`, compute the final mask from the number of remaining bits.

```text
remainingBits = min(RADIX_BITS, KEY_BITS - shift)
passMask = (1 << remainingBits) - 1
```

Do not hardcode the pass count independently from the key width and radix width.

# 3. Use the canonical three-stage architecture

Treat one radix pass as three logical stages:

```text
1. Per-workgroup histogram
2. Global prefix scan
3. Stable scatter
```

The complete sort is:

```text
for every digit from least significant to most significant:
    build histograms
    compute global offsets
    scatter stably
    swap input and output index buffers
```

This three-stage architecture is the portable baseline.

A concrete implementation may fuse stages later, but only after the unfused version is correct and tested.

# 4. Allocate the required buffers

Keep the key source immutable:

```text
keyBuffer[M]
```

Use two ping-pong buffers for the index permutation:

```text
indicesA[N]
indicesB[N]
```

Allocate scratch storage for the histogram and scan stages:

```text
histograms[NUM_GROUPS][NUM_BINS]
groupPrefixes[NUM_GROUPS][NUM_BINS]
binTotals[NUM_BINS]
binBases[NUM_BINS]
```

The exact physical layout may be either:

```text
histograms[group][bin]
```

or:

```text
histograms[bin][group]
```

Choose the layout based on the dominant memory access pattern of the scan stage. Keep the logical meaning independent from the physical layout.

At the start:

```text
inputIndices  = indicesA
outputIndices = indicesB
```

After every radix pass:

```text
swap(inputIndices, outputIndices)
```

Return the final `inputIndices` handle after the last swap. Do not assume that the result always resides in the original index buffer.

No ping-pong key buffers are required because every pass obtains the current key through:

```text
sourceIndex = inputIndices[position]
key = keyBuffer[sourceIndex]
```

# 5. Implement the histogram kernel

## Contract

For every workgroup and every bin, compute:

```text
histograms[groupId][bin]
```

where the value is the number of index entries in that workgroup's input range whose indirectly referenced key has the current radix digit equal to `bin`.

Each workgroup owns one continuous range of positions in the current index permutation:

```text
groupBegin = groupId * ITEMS_PER_GROUP
```

Assign positions in a deterministic blocked layout:

```text
position = groupBegin
         + itemIndex * WORKGROUP_SIZE
         + localThreadId
```

The position identifies an entry in `inputIndices`, not an entry in `keyBuffer`.

## Required algorithm

Inside each workgroup:

1. Allocate `localHistogram[NUM_BINS]` in shared memory.
2. Clear it cooperatively.
3. Execute a workgroup barrier.
4. Let every thread process `ITEMS_PER_THREAD` index positions.
5. Read `sourceIndex = inputIndices[position]`.
6. Read and encode `keyBuffer[sourceIndex]` into a sortable unsigned key.
7. Extract the current digit.
8. Increment `localHistogram[digit]` atomically.
9. Execute a workgroup barrier.
10. Copy the local histogram to the global histogram buffer.

## Abstract pseudocode

```cpp
kernel histogram(
    ReadOnlyStorageBuffer keyBuffer,
    ReadOnlyStorageBuffer inputIndices,
    StorageBuffer histograms,
    uint N,
    uint shift,
    uint passMask
) {
    shared atomic_uint localHistogram[NUM_BINS];

    for (uint bin = localThreadId;
         bin < NUM_BINS;
         bin += WORKGROUP_SIZE) {
        atomicStore(localHistogram[bin], 0);
    }

    workgroupBarrier();

    uint groupBegin = groupId * ITEMS_PER_GROUP;

    for (uint item = 0; item < ITEMS_PER_THREAD; ++item) {
        uint position = groupBegin
                      + item * WORKGROUP_SIZE
                      + localThreadId;

        if (position < N) {
            Index sourceIndex = inputIndices[position];
            KeyValue value = keyBuffer[sourceIndex];
            UnsignedKey key = encodeSortableKey(value);
            uint digit = uint((key >> shift) & passMask);
            atomicAdd(localHistogram[digit], 1);
        }
    }

    workgroupBarrier();

    for (uint bin = localThreadId;
         bin < NUM_BINS;
         bin += WORKGROUP_SIZE) {
        histograms[groupId][bin] = atomicLoad(localHistogram[bin]);
    }
}
```

The ordering of histogram atomics is irrelevant because this kernel computes counts only.

# 6. Implement the global prefix scan

The scan stage must produce the starting output range for every `(group, bin)` pair.

For each bin, compute an exclusive scan over workgroups:

```text
groupPrefixes[group][bin]
    = sum(histograms[previousGroup][bin])
```

Also compute:

```text
binTotals[bin]
    = sum(histograms[allGroups][bin])
```

Then compute an exclusive scan over bins:

```text
binBases[bin]
    = sum(binTotals[previousBin])
```

The final start offset for one workgroup and one bin is:

```text
groupOffset(group, bin)
    = binBases[bin]
    + groupPrefixes[group][bin]
```

This offset guarantees both:

- bins occupy non-overlapping global output ranges;
- earlier workgroups write before later workgroups for the same bin.

## Reusable hierarchical scan

Implement scan as a reusable GPU primitive.

For an input larger than one workgroup:

1. Scan independent blocks.
2. Store each block total in `blockSums`.
3. Recursively scan `blockSums`.
4. Add the scanned block offsets back to every block.

Conceptually:

```text
input
  -> scan blocks
  -> partial output + block sums
  -> scan block sums
  -> add block offsets
  -> complete exclusive scan
```

The scan of `binTotals` commonly fits into one workgroup when `NUM_BINS` is 16 or 256.

The scan across workgroups may require multiple dispatches.

Treat “prefix scan” as one logical stage even when it is implemented by several kernels.

# 7. Implement stable index scatter

## Required output index

For each position in the current index permutation:

```text
outputPosition
    = binBases[digit]
    + groupPrefixes[groupId][digit]
    + processedEarlierTiles[digit]
    + localRank
```

Where:

- `binBases[digit]` is the global start of the digit's range;
- `groupPrefixes[groupId][digit]` skips equal digits from earlier workgroups;
- `processedEarlierTiles[digit]` skips equal digits from earlier tiles in the same workgroup;
- `localRank` counts equal digits earlier in the current tile.

The scatter writes only the referenced index:

```text
outputIndices[outputPosition] = sourceIndex
```

It does not write or reorder the key value.

## Tile model

A workgroup processes `ITEMS_PER_THREAD` tiles sequentially.

Each tile contains `WORKGROUP_SIZE` positions from the current index permutation:

```text
tile 0: groupBegin + [0, WORKGROUP_SIZE)
tile 1: groupBegin + [WORKGROUP_SIZE, 2 * WORKGROUP_SIZE)
...
```

Processing tiles in increasing permutation order is part of the stability guarantee.

## Stable local rank

For an index entry at local position `p`, define:

```text
localRank
    = number of positions q where
      q < p and tileDigit[q] == tileDigit[p]
```

Do not use an unconstrained atomic increment to assign ranks:

```cpp
rank = atomicAdd(counter[digit], 1); // not guaranteed stable
```

Atomic execution order does not have to match local thread order.

## Portable baseline for local rank

Use shared memory:

1. Every thread resolves its current index to a key and writes the resulting digit to `tileDigits[localThreadId]`.
2. Execute a workgroup barrier.
3. Every valid thread loops over previous positions and counts equal digits.

This baseline is not optimal, but it is portable and stable.

## Abstract pseudocode

```cpp
kernel scatterIndices(
    ReadOnlyStorageBuffer keyBuffer,
    ReadOnlyStorageBuffer inputIndices,
    StorageBuffer outputIndices,
    StorageBuffer groupPrefixes,
    StorageBuffer binBases,
    uint N,
    uint shift,
    uint passMask
) {
    shared uint processedEarlierTiles[NUM_BINS];
    shared atomic_uint tileCounts[NUM_BINS];
    shared uint tileDigits[WORKGROUP_SIZE];

    for (uint bin = localThreadId;
         bin < NUM_BINS;
         bin += WORKGROUP_SIZE) {
        processedEarlierTiles[bin] = 0;
    }

    workgroupBarrier();

    uint groupBegin = groupId * ITEMS_PER_GROUP;

    for (uint tile = 0; tile < ITEMS_PER_THREAD; ++tile) {
        for (uint bin = localThreadId;
             bin < NUM_BINS;
             bin += WORKGROUP_SIZE) {
            atomicStore(tileCounts[bin], 0);
        }

        uint position = groupBegin
                      + tile * WORKGROUP_SIZE
                      + localThreadId;

        bool valid = position < N;
        Index sourceIndex = 0;
        uint digit = INVALID_DIGIT;

        if (valid) {
            sourceIndex = inputIndices[position];
            KeyValue value = keyBuffer[sourceIndex];
            UnsignedKey key = encodeSortableKey(value);
            digit = uint((key >> shift) & passMask);
            tileDigits[localThreadId] = digit;
        } else {
            tileDigits[localThreadId] = INVALID_DIGIT;
        }

        workgroupBarrier();

        if (valid) {
            atomicAdd(tileCounts[digit], 1);
        }

        workgroupBarrier();

        if (valid) {
            uint localRank = 0;

            for (uint previous = 0;
                 previous < localThreadId;
                 ++previous) {
                if (tileDigits[previous] == digit) {
                    ++localRank;
                }
            }

            uint outputPosition =
                binBases[digit]
                + groupPrefixes[groupId][digit]
                + processedEarlierTiles[digit]
                + localRank;

            outputIndices[outputPosition] = sourceIndex;
        }

        workgroupBarrier();

        for (uint bin = localThreadId;
             bin < NUM_BINS;
             bin += WORKGROUP_SIZE) {
            processedEarlierTiles[bin] += atomicLoad(tileCounts[bin]);
        }

        workgroupBarrier();
    }
}
```

Both histogram and scatter must derive the radix digit from exactly the same expression:

```text
encodeSortableKey(keyBuffer[inputIndices[position]])
```

# 8. Preserve stability at every hierarchy level

Verify stability separately at three levels.

## Across workgroups

`groupPrefixes[group][bin]` must include all equal-digit elements from previous workgroups.

Therefore workgroup 0 writes before workgroup 1 for the same digit.

## Across tiles

Tiles must be processed in increasing order of positions in the current input index buffer.

`processedEarlierTiles[bin]` must include all equal-digit elements from previous tiles.

## Within one tile

`localRank` must count only matching elements at lower local positions.

If all three conditions hold, the resulting index permutation is globally stable.

# 9. Orchestrate passes on the host

Use API-independent orchestration similar to:

```cpp
uint numPasses = ceilDiv(KEY_BITS, RADIX_BITS);

Buffer inputIndices = indicesA;
Buffer outputIndices = indicesB;

for (uint pass = 0; pass < numPasses; ++pass) {
    uint shift = pass * RADIX_BITS;
    uint remainingBits = min(RADIX_BITS, KEY_BITS - shift);
    uint passMask = (1u << remainingBits) - 1u;

    dispatchHistogram(
        keyBuffer,
        inputIndices,
        histograms,
        N,
        shift,
        passMask
    );

    makeStorageWritesVisibleToNextDispatch();

    dispatchGroupHistogramScan(
        histograms,
        groupPrefixes,
        binTotals,
        NUM_GROUPS,
        NUM_BINS
    );

    makeStorageWritesVisibleToNextDispatch();

    dispatchBinTotalsScan(
        binTotals,
        binBases,
        NUM_BINS
    );

    makeStorageWritesVisibleToNextDispatch();

    dispatchScatterIndices(
        keyBuffer,
        inputIndices,
        outputIndices,
        groupPrefixes,
        binBases,
        N,
        shift,
        passMask
    );

    makeStorageWritesVisibleToNextDispatch();

    swap(inputIndices, outputIndices);
}

return inputIndices;
```

`keyBuffer` is bound read-only for every pass and never changes roles.

The concrete API may combine multiple ordered dispatches inside one command buffer or compute pass when allowed.

The algorithmic requirement is visibility and ordering, not a specific command-buffer structure.

# 10. Isolate the API adapter

Keep the following operations behind an API-specific adapter:

```text
createStorageBuffer(size, usage)
createComputePipeline(shader, constants)
createBindings(resources)
setPushConstantsOrUniforms(parameters)
dispatch(pipeline, groupCount)
insertStorageDependency()
submit()
readBackForValidation()
```

The algorithm layer should not contain WebGPU, Metal, Vulkan, or CUDA-specific names.

## WebGPU adapter notes

Typical mappings:

```text
keyBuffer/inputIndices -> read-only storage bindings
outputIndices/scratch  -> read-write storage bindings
buffers                    -> GPUBuffer with STORAGE usage
kernel                     -> compute pipeline
bindings                   -> bind groups
workgroup shared memory    -> var<workgroup>
workgroup barrier          -> workgroupBarrier()
dispatch                   -> dispatchWorkgroups()
```

Commands recorded in order in one command encoder are ordered, but the implementation must still respect WebGPU's usage and pass rules. Split compute passes when necessary for resource usage transitions or implementation clarity.

## Metal adapter notes

Typical mappings:

```text
keyBuffer/inputIndices -> MTLBuffer arguments read by the kernel
outputIndices/scratch  -> MTLBuffer arguments written by the kernel
kernel                     -> MTLComputePipelineState
bindings                   -> setBuffer / argument buffers
workgroup shared memory    -> threadgroup address space
dispatch                   -> dispatchThreadgroups or dispatchThreads
```

Use separate dispatches for stages that require synchronization across threadgroups. A threadgroup barrier does not synchronize different threadgroups.

These notes are examples only. Keep the sorting logic independent from them.

# 11. Never assume a global barrier inside one dispatch

A workgroup or threadgroup barrier synchronizes only threads in the same group.

It cannot make one workgroup wait for all other workgroups.

Therefore operations with a global dependency must be separated into ordered dispatches unless a mathematically valid decoupled look-back or equivalent algorithm is intentionally implemented.

The portable baseline must use separate dispatches for:

```text
histogram writes -> scan reads
scan writes      -> scatter reads
scatter writes   -> next pass histogram reads
```

# 12. Transform indirectly referenced key types

## Signed integers

When `keyBuffer[sourceIndex]` is signed, map it to an unsigned sortable representation by flipping the sign bit:

```cpp
sortable = bitcastUnsigned(value) ^ SIGN_BIT;
```

For a 32-bit signed integer:

```text
SIGN_BIT = 0x80000000
```

The same XOR restores the original representation if required.

## IEEE-754 floating-point values

When `keyBuffer[sourceIndex]` is IEEE-754 floating point, use a monotonic bit transform:

```cpp
bits = bitcastUnsigned(value);

if (signBitIsSet(bits)) {
    sortable = bitwiseNot(bits);
} else {
    sortable = bits ^ SIGN_BIT;
}
```

Define a policy for NaN values before implementing floating-point sorting.

Possible policies include:

- reject NaNs;
- place NaNs first;
- place NaNs last;
- preserve NaN payload order.

# 13. Build a correct baseline before optimizing

Recommended initial configuration:

```text
RADIX_BITS       = 4 or 8
WORKGROUP_SIZE   = 128 or 256
ITEMS_PER_THREAD = 1 to 8
```

Implement first:

```text
histogram kernel
hierarchical exclusive scan
portable stable index scatter
index ping-pong orchestration
CPU validation
```

Do not begin with fused kernels, vendor-specific subgroup assumptions, or a decoupled look-back scan.

# 14. Optimize in a controlled order

Apply optimizations only after correctness tests pass.

## Optimization 1: process multiple items per thread

Increasing `ITEMS_PER_THREAD`:

- reduces workgroup count;
- reduces histogram scratch size;
- reduces scheduling overhead;
- may reduce occupancy and parallelism when too large.

Benchmark several values.

## Optimization 2: replace the portable local-rank loop

Possible faster implementations:

- subgroup ballot plus population count;
- subgroup prefix operations;
- shared bit masks;
- workgroup-wide segmented scan;
- vendor-provided SIMD-group primitives.

The optimized implementation must preserve the same local-rank contract.

## Optimization 3: optimize scan

Possible scan implementations:

- Blelloch scan;
- Hillis-Steele scan;
- subgroup scan plus shared subgroup totals;
- hierarchical block scan;
- decoupled look-back, when supported and thoroughly validated.

## Optimization 4: tune radix width

Larger `RADIX_BITS` means:

```text
fewer passes
more bins
more shared memory
larger histogram scratch
more expensive scans
```

Smaller `RADIX_BITS` means:

```text
more passes
fewer bins
less shared memory
smaller scans
```

Benchmark at least 4-bit and 8-bit digits for the target hardware and data size.

## Optimization 5: fuse stages carefully

Possible fusions include:

- scan of bin totals with creation of bin bases;
- offset calculation with stable index scatter;
- multiple local tiles in one histogram kernel.

Do not fuse stages that require a global synchronization point unless the replacement algorithm explicitly removes that dependency.

# 15. Optional two-kernel architecture

A specialized implementation may use only:

```text
Kernel 1: per-workgroup histogram
Kernel 2: offset calculation plus stable index scatter
```

The second kernel can read the histograms of all workgroups and derive its own offsets before scattering.

It must still read keys indirectly and write only indices:

```text
sourceIndex = inputIndices[position]
key = keyBuffer[sourceIndex]
outputIndices[outputPosition] = sourceIndex
```

Advantages:

- fewer dispatches;
- fewer intermediate buffers;
- simpler host scheduling.

Disadvantages:

- repeated reads of the same histogram data;
- work proportional to the number of workgroups inside each workgroup;
- weaker scaling for very large workgroup counts.

Treat this as an optimization for a measured workload, not as the default portable design.

# 16. Validate correctness

Compare the GPU index permutation with a CPU stable sort of the same input index sequence using the indirect key lookup.

## Indirect sorted-order check

```cpp
for (uint i = 1; i < N; ++i) {
    KeyValue previous = keyBuffer[sortedIndices[i - 1]];
    KeyValue current = keyBuffer[sortedIndices[i]];
    assert(previous <= current);
}
```

## Permutation check

Verify that `sortedIndices` contains exactly the same index entries as `inputIndices`, including multiplicities if repeated indices are permitted.

The sort must not create, remove, or modify index values.

## Stability check

Use an input index sequence whose referenced keys contain duplicates:

```text
keyBuffer:    [5, 2, 5, 5, 2]
inputIndices: [0, 1, 2, 3, 4]
```

Expected stable result:

```text
sortedIndices: [1, 4, 0, 2, 3]
```

Because:

```text
keyBuffer[1] = 2
keyBuffer[4] = 2
keyBuffer[0] = 5
keyBuffer[2] = 5
keyBuffer[3] = 5
```

For equal referenced keys, positions from the original input index sequence must remain in their original relative order.

Also test a non-identity initial permutation, for example:

```text
inputIndices: [3, 0, 4, 2, 1]
```

Stability must be defined relative to this sequence, not relative to numeric index order.

## Required test sizes

Test at least:

```text
N = 0
N = 1
N < WORKGROUP_SIZE
N = WORKGROUP_SIZE
N = WORKGROUP_SIZE + 1
N = ITEMS_PER_GROUP - 1
N = ITEMS_PER_GROUP
N = ITEMS_PER_GROUP + 1
N spanning several workgroups
large N
```

## Required data patterns

Test:

```text
random referenced keys
already sorted index permutations
reverse-sorted index permutations
all referenced keys equal
two distinct key values
many duplicate referenced keys
minimum and maximum key values
non-identity input permutations
subsets of the key buffer
sizes not divisible by the workgroup size
sizes not divisible by ITEMS_PER_GROUP
```

# 17. Diagnose common failures

## Indirect keys are not sorted

Likely causes:

- reading `keyBuffer[position]` instead of `keyBuffer[inputIndices[position]]`;
- histogram and scatter use different key-encoding logic;
- reading from the original index buffer on every pass instead of the current ping-pong input;
- incorrect digit shift or pass mask.

## Output is sorted but unstable

Likely causes:

- atomic increment used directly as local rank;
- workgroups do not receive offsets in current input-permutation order;
- tiles are processed in a non-deterministic order;
- stability is incorrectly checked against numeric index order instead of input index order.

## Some indices disappear or are duplicated

Likely causes:

- incorrect exclusive scan;
- incorrect histogram buffer indexing;
- missing bounds checks;
- missing storage-write visibility between dispatches;
- local counters not reset between tiles;
- scatter writes a position or key instead of the original `sourceIndex`.

## Correct for one workgroup but wrong for several

Likely causes:

- assuming a workgroup barrier is global;
- missing group prefixes;
- incorrect hierarchical scan;
- overwriting shared scratch before all threads finish reading it.

## Correct only for aligned input sizes

Likely causes:

- missing `position < N` checks;
- invalid threads included in histogram or local rank;
- final partial tile not represented consistently.

## Result appears in the wrong index buffer

Likely cause:

- fixed output-buffer assumption instead of tracking the ping-pong swap.

## The key buffer changed

Likely cause:

- the implementation accidentally bound the key buffer as writable or reused it as ping-pong storage. The key buffer must remain immutable throughout the sort.

# 18. Required implementation deliverables

A complete implementation should contain:

```text
1. IndirectRadixSortConfig
2. scratch-size calculator
3. index ping-pong allocator or caller-provided output-index contract
4. histogram kernel using indirect key lookup
5. reusable exclusive-scan implementation
6. stable index-scatter kernel
7. host-side pass scheduler
8. API adapter for bindings and synchronization
9. key encoding helpers
10. CPU reference, permutation, and stability tests
11. benchmark harness
```

Recommended public interface:

```cpp
struct IndirectRadixSortConfig {
    uint keyBits;
    uint radixBits;
    uint workgroupSize;
    uint itemsPerThread;
};

struct IndirectRadixSortResult {
    Buffer sortedIndices;
};

IndirectRadixSortResult radixSortIndices(
    ComputeContext context,
    Buffer keyBuffer,
    Buffer inputIndices,
    uint numIndices,
    IndirectRadixSortConfig config
);
```

The semantic contract is:

```text
Sort inputIndices by the values of keyBuffer[inputIndices[position]].
Return the reordered index buffer.
Do not reorder keyBuffer.
```

Hide the following implementation details from callers:

- index ping-pong buffer selection;
- histogram scratch;
- scan scratch;
- number of internal dispatches;
- API-specific barriers;
- subgroup optimization selection.

# 19. Definition of done

The implementation is complete when all of the following are true:

1. The pass count is derived from encoded key width and radix width.
2. Every workgroup histogram is computed from `keyBuffer[inputIndices[position]]`.
3. The global exclusive scan produces non-overlapping output ranges.
4. Scatter reorders only index values and is stable.
5. Partial workgroups and partial tiles are handled safely.
6. Index ping-pong buffers work for both odd and even pass counts.
7. The final index buffer references the same entries as the input index buffer.
8. Indirectly referenced keys are in sorted order.
9. Equal referenced keys preserve their order from the input index sequence.
10. The key buffer remains unchanged.
11. No algorithmic step depends on workgroup execution order.
12. Subgroup features improve performance but are not required for correctness.
13. API-specific code is confined to the adapter layer.
14. The implementation passes duplicate-heavy, non-identity-permutation, subset, and boundary-size tests.

# 20. Mental model

An indirect GPU radix sort repeatedly transforms an index permutation while leaving the key source untouched:

```text
Read each key through the current index buffer.
    ->
Count how many referenced keys belong to every digit bucket.
    ->
Compute where every bucket and workgroup range begins.
    ->
Move only the indices to those positions while preserving input order.
    ->
Swap the input and output index buffers.
    ->
Process the next more-significant digit.
```

In compact form:

```cpp
for each radix digit from least significant to most significant:
    histogram(keyBuffer, inputIndices)
    exclusiveScan(histograms)
    stableScatterIndices(keyBuffer, inputIndices, outputIndices)
    swap(inputIndices, outputIndices)
```

The defining access pattern is:

```text
sourceIndex = inputIndices[position]
key = keyBuffer[sourceIndex]
```

The defining write is:

```text
outputIndices[outputPosition] = sourceIndex
```

Everything else—radix width, workgroup size, subgroup operations, memory layout, scan implementation, and kernel fusion—is an optimization of this core algorithm.
