# GPU radix sort experiments

The first implementation is a stable indirect LSD radix sort for three.js and
WebGPU. It sorts an index buffer by unsigned 32-bit values read from an immutable
key buffer (`keys[indices[position]]`). The compute kernels are plain WGSL strings;
Three.js supplies the renderer and owns the input/output storage buffers, while
the sorter records the WebGPU compute pass directly. The demo and implementation
are written in TypeScript and built with Vite.

```js
import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { RadixSort } from './src/RadixSort';

const keys = new StorageBufferAttribute(Uint32Array.of(40, 10, 30), 1);
const indices = new StorageBufferAttribute(Uint32Array.of(2, 0, 1), 1);
const sortedIndices = new StorageBufferAttribute(new Uint32Array(3), 1);
const sorter = new RadixSort(renderer, keys, indices, sortedIndices);

await sorter.sortAsync(); // sortedIndices now contains [ 1, 2, 0 ]
sorter.dispose();
```

`sortedIndices` is owned by the caller: the sorter never replaces it, so the
same storage attribute can be bound directly by a 3DGS rendering pipeline.
`sortAsync()` waits for the submitted GPU work and is useful for readback,
validation, and timing. In a render loop, call `sort()` instead; it submits the
same commands without a CPU/GPU synchronization point. Call `dispose()` when the
sorter is no longer needed to release its private scratch buffers.

The implementation uses eight stable four-bit passes. One GPU invocation owns a
contiguous 256-item block and walks it in input order. Each pass builds private
block histograms without atomics, scans block histograms and digit totals, and
then performs a stable block-local scatter without the previous quadratic rank
search. Histogram and scatter use `dispatchWorkgroupsIndirect()`; the dispatch
buffer also stores the exact item count used by the shaders.

## Demo

```sh
npm install
npm run dev
```

Open the shown local URL in a WebGPU-capable browser. The demo first warms up the
radix pipelines, then runs isolated experiments at 10, 100, 1,000, 10,000,
100,000, and 1,000,000 elements. Every experiment creates deterministic,
duplicate-heavy keys and shuffled indices, times both implementations, reads the
GPU result back, and compares every index with stable JavaScript `Array.sort()`
and the CPU hybrid radix implementation exported by Three.js `SortUtils`. Despite
the occasional “bitonic” label, `three/addons/utils/SortUtils.js` currently
exports `radixSort`, not a bitonic GPU sorter. The results table reports all
three timings and two GPU speedup ratios relative to the CPU baselines. Ratios
divide CPU time by GPU time: above `1×` means GPU radix is faster and below `1×`
means it is slower (for example, 12.12 ms / 3.50 ms = 3.46×).
Short CPU sorts are repeated for at least 50 ms and displayed as average time
per operation, avoiding zero-duration samples and infinite percentages.

## Tests

```sh
npm test
npm run typecheck
npm run format:check
npm run build
```

The Vitest tests cover the CPU oracle used by the browser demo, including
stability, indirect lookup, empty input, and repeated indices.
