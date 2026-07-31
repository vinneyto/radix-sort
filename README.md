# GPU radix sort experiments

The first implementation is a stable indirect LSD radix sort for three.js and
WebGPU. It sorts an index buffer by unsigned 32-bit values read from an immutable
key buffer (`keys[indices[position]]`) and uses TSL for all compute kernels.
The demo and implementation are written in TypeScript and built with Vite.

```js
import { StorageBufferAttribute } from 'three';
import WebGPURenderer from 'three/addons/renderers/webgpu/WebGPURenderer.js';
import { RadixSort } from './src/RadixSort';

const keys = new StorageBufferAttribute( Uint32Array.of( 40, 10, 30 ), 1 );
const indices = new StorageBufferAttribute( Uint32Array.of( 2, 0, 1 ), 1 );
const sortedIndices = new StorageBufferAttribute( new Uint32Array( 3 ), 1 );
const sorter = new RadixSort( renderer, keys, indices, sortedIndices );

await sorter.sortAsync(); // sortedIndices now contains [ 1, 2, 0 ]
```

`sortedIndices` is owned by the caller: the sorter never replaces it, so the
same storage attribute can be bound directly by a 3DGS rendering pipeline.
`sortAsync()` waits for completion and is useful for readback and validation.
In a render loop, call `sort()` instead; it enqueues the same passes through
`renderer.compute()` without waiting for CPU/GPU synchronization, and later
render commands can consume `sortedIndices` in submission order.

The baseline uses eight stable four-bit passes. Each pass dispatches a
per-workgroup histogram, a global offset scan, and a stable scatter. The simple
scatter favors portability and correctness over peak performance.

## Demo

```sh
npm install
npm run dev
```

Open the shown local URL in a WebGPU-capable browser. The demo creates duplicate
keys and a reverse index permutation, runs the GPU sort, reads the result back,
and checks every index against a stable JavaScript `Array.sort()` reference.

## Tests

```sh
npm test
npm run typecheck
npm run build
```

The Vitest tests cover the CPU oracle used by the browser demo, including
stability, indirect lookup, empty input, and repeated indices.
