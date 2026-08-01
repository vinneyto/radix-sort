import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { radixSort as threeRadixSort } from 'three/addons/utils/SortUtils.js';
import { RadixSort } from './RadixSort';
import { referenceSort } from './reference';
import './style.css';

const SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000] as const;
const WARMUP_SIZE = 4_096;

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
	await sorter.sortAsync();
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

function threeSort(keys: Uint32Array, indices: Uint32Array): { result: Uint32Array; milliseconds: number } {
	const result = indices.slice();
	const started = performance.now();
	sortWithThree(result, { aux: new Uint32Array(result.length), get: (index) => keys[index] });
	return { result, milliseconds: performance.now() - started };
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
		const cpuStarted = performance.now();
		const expected = referenceSort(keys, indices);
		const cpuMilliseconds = performance.now() - cpuStarted;
		const three = threeSort(keys, indices);
		const gpu = await radixSort(renderer, keys, indices);
		assertEqual(expected, three.result);
		assertEqual(expected, gpu.result);

		const row = body.insertRow();
		row.insertCell().textContent = count.toLocaleString();
		row.insertCell().textContent = formatTime(gpu.milliseconds);
		row.insertCell().textContent = formatTime(three.milliseconds);
		row.insertCell().textContent = formatTime(cpuMilliseconds);
		row.insertCell().textContent = `${((gpu.milliseconds / three.milliseconds) * 100).toFixed(1)}%`;
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
