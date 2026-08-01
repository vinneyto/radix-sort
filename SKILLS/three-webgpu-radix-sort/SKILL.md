---
name: three-webgpu-radix-sort
description: Implement, port, debug, validate, and benchmark stable indirect radix sorting in TypeScript with Three.js WebGPURenderer, TSL resource nodes, native WGSL helpers through wgslFn, workgroup memory, atomics, storage buffers, and GPU readback. Use for Three.js WebGPU compute shader validation errors, invalid pipelines, wgslFn integration, radix-sort correctness comparisons against Array.sort, pipeline warmup, or generated-WGSL inspection.
---

# Three.js WebGPU radix sort

Use `../radix-impl/SKILL.md` for the API-independent algorithm and stability proof. Apply the Three.js/WGSL integration rules below when implementing that algorithm in TypeScript.

## Preserve the indirect stable-sort contract

- Keep keys immutable.
- Read `sourceIndex = inputIndices[position]` and then `key = keys[sourceIndex]` in both histogram and scatter.
- Ping-pong indices only.
- Process radix digits least-significant first.
- Give each `(workgroup, digit)` a globally exclusive offset.
- Compute the within-tile rank from lower local positions; never use atomic execution order as a stable rank.
- Handle partial workgroups with a validity flag and a sentinel digit.

Start with the portable three-dispatch pass:

1. workgroup histogram;
2. global offset scan;
3. stable scatter.

Prefer correctness over subgroup or fused-kernel optimizations until full-array comparisons pass.

## Divide responsibilities between TSL and WGSL

Use TSL only to create and bind resources:

- `storage(...)` for storage buffer attributes;
- `uniform(...)` for count, shift, and group count;
- `workgroupArray(...)` for shared arrays;
- `globalId`, `localId`, and `workgroupId` for built-ins;
- `.compute(invocationCount, [workgroupSize])` for dispatch nodes.

Put loops, barriers, atomics, digit extraction, scan, and scatter logic in named `wgslFn` functions.

Construct a compute node **directly from the native function call**:

```ts
const kernel = wgslFn(`
  fn kernel_name(
    data: ptr<storage, array<u32>, read_write>,
    scratch: ptr<workgroup, array<u32, 256>>,
    gid: u32
  ) -> void {
    (*scratch)[gid] = (*data)[gid];
  }
`);

const scratch = workgroupArray('uint', 256);
const computeNode = kernel({
  data,
  scratch,
  gid: localId.x
}).compute(invocationCount, [256]);
```

Do not wrap a void `wgslFn` call in an otherwise empty inline `Fn`. Do not rely on deprecated `.append()` to emit the call. These patterns may register the helper and bindings while leaving compute `main` without the actual helper invocation.

## Declare WGSL pointers exactly

Specify an access mode only for `storage` pointers:

```wgsl
ptr<storage, array<u32>, read>
ptr<storage, array<u32>, read_write>
```

Do **not** specify `read_write` on `workgroup` pointers. WGSL rejects it because only the storage address space accepts an explicit access mode:

```wgsl
// Correct
ptr<workgroup, array<u32, 256>>
ptr<workgroup, array<atomic<u32>, 16>>

// Invalid
ptr<workgroup, array<u32, 256>, read_write>
```

Declare a shared histogram as atomic at the type level:

```ts
const localHistogram = workgroupArray('atomic<u32>', RADIX_SIZE);
```

Then use atomic pointers in WGSL:

```wgsl
atomicStore(&(*local_histogram)[bin], 0u);
atomicAdd(&(*local_histogram)[digit], 1u);
let count = atomicLoad(&(*local_histogram)[bin]);
```

A plain `array<u32>` element is not a valid operand for `atomicAdd`, even if it lives in workgroup memory.

## Respect barriers and dispatch boundaries

- Place `workgroupBarrier()` after cooperative initialization and before consuming shared writes.
- Ensure every invocation reaches each barrier through uniform control flow.
- Keep histogram, global scan, and scatter in separate ordered dispatches; a workgroup barrier is not global.
- In async validation code, await dispatch completion before readback.
- Track the final ping-pong destination. For eight four-bit passes, deliberately arrange the buffers so the final pass lands in the caller-owned output attribute.

## Keep GPU resources reachable

Create storage attributes before building nodes and retain them through node closures or class fields. Ensure the caller-owned output is referenced by an emitted scatter call; otherwise Three.js may never create its GPU buffer and `getArrayBufferAsync(outputAttribute)` can fail while reading backend metadata.

Validate constructor contracts early:

- scalar `itemSize` for keys and indices;
- distinct input and output attributes;
- requested length within both capacities;
- valid indices into the key buffer.

## Inspect generated WGSL before guessing

When pipelines fail, find the first shader-module parsing or validation error. Later `Invalid ComputePipeline` and `Invalid CommandBuffer` messages are cascades, not separate causes.

Generate each compute node with the same Three.js `WGSLNodeBuilder` used by the active renderer and inspect `computeShader`. Verify all of the following:

- `main` contains the `radix_histogram(...)`, `radix_scan(...)`, or `radix_scatter(...)` call;
- storage bindings have the intended access modes;
- workgroup pointer types omit access modes;
- the histogram array is `array<atomic<u32>, N>`;
- each helper receives pointers such as `&NodeBuffer.value` and `&WorkgroupArray`;
- workgroup size and built-in parameters are correct.

Use the same Three.js module instance for node construction and the builder. Mixing source and bundled copies can produce separate global TSL stacks and misleading empty shaders.

Run an independent WGSL validator such as Tint or Naga when available. TypeScript checking and Vite builds do not parse or validate WGSL strings.

## Cross-check against Array.sort

Use deterministic, isolated experiments. For each size:

1. create fresh keys and a fresh shuffled index permutation;
2. include many duplicate keys to exercise stability;
3. copy the same logical inputs to both implementations;
4. time stable JavaScript `Array.sort()`;
5. time the completed radix dispatches separately from GPU readback;
6. read back the entire GPU result;
7. compare every index in order and report the first mismatch;
8. record size, CPU time, GPU time, ratio, and pass/fail status.

Use the direct stable oracle:

```ts
const expected = Array.from(indices).sort((a, b) => keys[a] - keys[b]);
```

Warm up with a separate untimed radix sort before measurements. Wait for it to complete so shader compilation and initial pipeline creation do not contaminate the first result. Use representative sizes from tiny inputs through the target maximum, including workgroup and multi-workgroup boundaries.

State timing boundaries clearly. In particular, distinguish:

- GPU dispatch completion time;
- GPU-to-CPU readback time;
- input generation and upload time;
- `Array.from` plus `Array.sort` time.

Never claim a performance advantage from a correctness-only run whose timing boundaries differ.

## Diagnose recurring failures

| Error or symptom | Likely cause | Correction |
| --- | --- | --- |
| `atomicAdd(ptr<workgroup, u32...>)` | shared element is plain `u32` | declare `workgroupArray('atomic<u32>', ...)` |
| only storage pointers may specify access mode | `read_write` supplied to `ptr<workgroup,...>` | omit the third pointer parameter |
| helper exists but compute `main` is empty | void `wgslFn` hidden behind `Fn`/`.append()` | call `wgslFn({...}).compute(...)` directly |
| output readback accesses undefined backend data | emitted shader never references output | verify scatter invocation in generated `main` |
| correct for one group only | missing global group prefix or assumed global barrier | use ordered scan dispatch and group offsets |
| sorted but unstable | atomic rank or wrong hierarchy order | rank by lower local positions; preserve group/tile order |
| missing or duplicated indices | bad exclusive offsets, bounds, or ping-pong source | compare entire output and inspect first mismatch |

## Completion checklist

Before reporting completion:

- run formatting, TypeScript checking, unit tests, and production build;
- inspect the generated WGSL for every distinct compute stage;
- validate WGSL with a real parser when available;
- run the WebGPU demo in a capable browser;
- warm up before benchmark timing;
- compare full results against stable `Array.sort` at all requested sizes;
- report environment limitations separately from code failures.
