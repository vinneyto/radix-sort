import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { RadixSort } from './RadixSort';
import { referenceSort, validateSort } from './reference';
import './style.css';

function requireElement<T extends Element>(selector: string): T {
	const element = document.querySelector<T>(selector);
	if (element === null) throw new Error(`Missing ${selector} element.`);
	return element;
}

const output = requireElement<HTMLPreElement>('#output');

async function main() {
	if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');

	const count = 4099;
	const keys = Uint32Array.from({ length: count }, () => Math.floor(Math.random() * 64));
	const input = Uint32Array.from({ length: count }, (_, i) => count - i - 1);
	const keyAttribute = new StorageBufferAttribute(keys, 1);
	const indexAttribute = new StorageBufferAttribute(input, 1);
	const outputAttribute = new StorageBufferAttribute(new Uint32Array(count), 1);
	const renderer = new WebGPURenderer();
	await renderer.init();

	const sorter = new RadixSort(renderer, keyAttribute, indexAttribute, outputAttribute);
	await sorter.sortAsync();
	const result = new Uint32Array(await renderer.getArrayBufferAsync(outputAttribute));
	validateSort(keys, input, result);

	output.className = 'ok';
	output.textContent = `PASS — ${count} indices match Array.sort()\n\nfirst 24 indices:\n${result.slice(0, 24).join(', ')}\n\nfirst 24 keys:\n${Array.from(result.slice(0, 24), (i) => keys[i]).join(', ')}`;
	console.log('Array.sort reference', referenceSort(keys, input));
}

main().catch((error: unknown) => {
	output.className = 'error';
	output.textContent = `FAIL — ${error instanceof Error ? error.stack : String(error)}`;
	console.error(error);
});
