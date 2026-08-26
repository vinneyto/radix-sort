import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { radixSort as threeRadixSort } from 'three/addons/utils/SortUtils.js';
import { RadixSort } from './RadixSort';
import { speedupRatio } from './benchmark';
import { referenceSort } from './reference';
import './style.css';

const SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000] as const;
const WARMUP_SIZE = 4_096;
const MIN_CPU_SAMPLE_MS = 50;
const MAX_CPU_ITERATIONS = 10_000;

const sortWithThree = threeRadixSort as unknown as (
	indices: Uint32Array,
	options: { aux?: Uint32Array; get: (index: number) => number }
) => void;

function requireElement<T extends Element>(selector: string): T {
	const element = document.querySelector<T>(selector);
	if (element === null) throw new Error(`Missing ${selector} element.`);
	return element;
}

function createExperimentData(count: number, seed: number): { keys: Uint32Array; indices: Uint32Array } {
	let state = seed >>> 0;
	const random = () => {
		state ^= state << 13;
		state ^= state >>> 17;
		state ^= state << 5;
		return state >>> 0;
	};

	// A bounded key range deliberately creates duplicates, so full equality also
	// checks that every radix pass preserves Array.sort's stable ordering.
	const keyRange = Math.max(2, Math.min(65_536, Math.ceil(count / 4)));
	const keys = Uint32Array.from({ length: count }, () => random() % keyRange);
	const indices = Uint32Array.from({ length: count }, (_, index) => index);
	for (let index = count - 1; index > 0; index--) {
		const other = random() % (index + 1);
		const value = indices[index];
		indices[index] = indices[other];
		indices[other] = value;
	}
	return { keys, indices };
}

async function radixSort(
	renderer: WebGPURenderer,
	keys: Uint32Array,
	indices: Uint32Array
): Promise<{ result: Uint32Array; milliseconds: number }> {
	const keyAttribute = new StorageBufferAttribute(keys, 1);
	const indexAttribute = new StorageBufferAttribute(indices, 1);
	const outputAttribute = new StorageBufferAttribute(new Uint32Array(indices.length), 1);
	const sorter = new RadixSort(renderer, keyAttribute, indexAttribute, outputAttribute);

	const started = performance.now();
	sorter.sort();
	await renderer.waitForGPU();
	const milliseconds = performance.now() - started;
	const result = new Uint32Array(await renderer.getArrayBufferAsync(outputAttribute));
	return { result, milliseconds };
}

function assertEqual(expected: readonly number[], actual: Uint32Array): void {
	if (expected.length !== actual.length) {
		throw new Error(`Length differs: Array.sort=${expected.length}, radix=${actual.length}`);
	}
	for (let index = 0; index < expected.length; index++) {
		if (expected[index] !== actual[index]) {
			throw new Error(`First mismatch at ${index}: Array.sort=${expected[index]}, radix=${actual[index]}`);
		}
	}
}

function measureCpu<T>(operation: () => T): { result: T; milliseconds: number } {
	let result!: T;
	let iterations = 0;
	const started = performance.now();
	let elapsed: number;
	do {
		result = operation();
		iterations++;
		elapsed = performance.now() - started;
	} while (elapsed < MIN_CPU_SAMPLE_MS && iterations < MAX_CPU_ITERATIONS);
	return { result, milliseconds: elapsed / iterations };
}

function threeSort(keys: Uint32Array, indices: Uint32Array): { result: Uint32Array; milliseconds: number } {
	return measureCpu(() => {
		const result = indices.slice();
		sortWithThree(result, { aux: new Uint32Array(result.length), get: (index) => keys[index] });
		return result;
	});
}

function formatTime(milliseconds: number): string {
	return `${milliseconds.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ms`;
}

async function main() {
	if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');

	const status = requireElement<HTMLParagraphElement>('#status');
	const body = requireElement<HTMLTableSectionElement>('#results');
	const renderer = new WebGPURenderer();
	await renderer.init();

	status.textContent = `Warming up radix pipelines on ${WARMUP_SIZE.toLocaleString()} elements…`;
	const warmup = createExperimentData(WARMUP_SIZE, 0xc0ffee);
	await radixSort(renderer, warmup.keys, warmup.indices);
	threeSort(warmup.keys, warmup.indices);

	for (const [experiment, count] of SIZES.entries()) {
		status.textContent = `Experiment ${experiment + 1}/${SIZES.length}: ${count.toLocaleString()} elements…`;
		await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

		const { keys, indices } = createExperimentData(count, 0x9e3779b9 ^ count);
		const array = measureCpu(() => referenceSort(keys, indices));
		const three = threeSort(keys, indices);
		const gpu = await radixSort(renderer, keys, indices);
		assertEqual(array.result, three.result);
		assertEqual(array.result, gpu.result);

		const row = body.insertRow();
		row.insertCell().textContent = count.toLocaleString();
		row.insertCell().textContent = formatTime(gpu.milliseconds);
		row.insertCell().textContent = formatTime(three.milliseconds);
		row.insertCell().textContent = formatTime(array.milliseconds);
		row.insertCell().textContent = `${speedupRatio(array.milliseconds, gpu.milliseconds).toFixed(2)}×`;
		row.insertCell().textContent = `${speedupRatio(three.milliseconds, gpu.milliseconds).toFixed(2)}×`;
		const resultCell = row.insertCell();
		resultCell.textContent = 'PASS';
		resultCell.className = 'pass';
	}

	status.textContent = `Complete — all ${SIZES.length} full-array comparisons passed.`;
	status.className = 'ok';
}

main().catch((error: unknown) => {
	const status = requireElement<HTMLParagraphElement>('#status');
	status.className = 'error';
	status.textContent = `FAIL — ${error instanceof Error ? error.stack : String(error)}`;
	console.error(error);
});
