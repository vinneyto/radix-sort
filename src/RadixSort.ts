import { StorageBufferAttribute, type WebGPURenderer } from 'three/webgpu';

const RADIX_BITS = 4;
const RADIX_SIZE = 1 << RADIX_BITS;
const PASSES = 32 / RADIX_BITS;
const BLOCK_ITEMS = 256;
const WORKGROUP_SIZE = 256;

const histogramShader = /* wgsl */ `
override shift: u32;

struct DispatchArguments {
	workgroup_count_x: u32,
	workgroup_count_y: u32,
	workgroup_count_z: u32,
	count: u32,
}

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_histograms: array<u32>;
@group(0) @binding(3) var<storage, read> dispatch: DispatchArguments;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let block = global_id.x;
	let block_start = block * ${BLOCK_ITEMS}u;
	if (block_start >= dispatch.count) {
		return;
	}

	var histogram: array<u32, ${RADIX_SIZE}>;
	for (var digit = 0u; digit < ${RADIX_SIZE}u; digit++) {
		histogram[digit] = 0u;
	}

	let block_end = min(block_start + ${BLOCK_ITEMS}u, dispatch.count);
	for (var position = block_start; position < block_end; position++) {
		let source_index = input[position];
		let digit = (keys[source_index] >> shift) & ${RADIX_SIZE - 1}u;
		histogram[digit]++;
	}

	let output_start = block * ${RADIX_SIZE}u;
	for (var digit = 0u; digit < ${RADIX_SIZE}u; digit++) {
		block_histograms[output_start + digit] = histogram[digit];
	}
}
`;

const scanBlockHistogramsShader = /* wgsl */ `
struct DispatchArguments {
	workgroup_count_x: u32,
	workgroup_count_y: u32,
	workgroup_count_z: u32,
	count: u32,
}

@group(0) @binding(0) var<storage, read> block_histograms: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_prefixes: array<u32>;
@group(0) @binding(2) var<storage, read_write> digit_totals: array<u32>;
@group(0) @binding(3) var<storage, read> dispatch: DispatchArguments;

@compute @workgroup_size(${RADIX_SIZE})
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
	let digit = local_id.x;
	let block_count = (dispatch.count + ${BLOCK_ITEMS - 1}u) / ${BLOCK_ITEMS}u;
	var running = 0u;

	for (var block = 0u; block < block_count; block++) {
		let address = block * ${RADIX_SIZE}u + digit;
		block_prefixes[address] = running;
		running += block_histograms[address];
	}

	digit_totals[digit] = running;
}
`;

const scanDigitTotalsShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> digit_totals: array<u32>;
@group(0) @binding(1) var<storage, read_write> digit_offsets: array<u32>;

@compute @workgroup_size(1)
fn main() {
	var running = 0u;
	for (var digit = 0u; digit < ${RADIX_SIZE}u; digit++) {
		digit_offsets[digit] = running;
		running += digit_totals[digit];
	}
}
`;

const scatterShader = /* wgsl */ `
override shift: u32;

struct DispatchArguments {
	workgroup_count_x: u32,
	workgroup_count_y: u32,
	workgroup_count_z: u32,
	count: u32,
}

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read> block_prefixes: array<u32>;
@group(0) @binding(3) var<storage, read> digit_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<storage, read> dispatch: DispatchArguments;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let block = global_id.x;
	let block_start = block * ${BLOCK_ITEMS}u;
	if (block_start >= dispatch.count) {
		return;
	}

	var local_counts: array<u32, ${RADIX_SIZE}>;
	for (var digit = 0u; digit < ${RADIX_SIZE}u; digit++) {
		local_counts[digit] = 0u;
	}

	let block_end = min(block_start + ${BLOCK_ITEMS}u, dispatch.count);
	let prefix_start = block * ${RADIX_SIZE}u;
	for (var position = block_start; position < block_end; position++) {
		let source_index = input[position];
		let digit = (keys[source_index] >> shift) & ${RADIX_SIZE - 1}u;
		let destination = digit_offsets[digit]
			+ block_prefixes[prefix_start + digit]
			+ local_counts[digit];

		local_counts[digit]++;
		output[destination] = source_index;
	}
}
`;

interface WebGPUBackendAccess {
	isWebGPUBackend?: boolean;
	device: GPUDevice | null;
	createStorageAttribute(attribute: StorageBufferAttribute): void;
	get(attribute: StorageBufferAttribute): { buffer?: GPUBuffer };
}

interface PassResources {
	histogramPipeline: GPUComputePipeline;
	histogramBindGroup: GPUBindGroup;
	scatterPipeline: GPUComputePipeline;
	scatterBindGroup: GPUBindGroup;
}

export interface RadixSortOptions {
	/** Number of entries at the beginning of the index buffer to sort. */
	length?: number;
}

/** Stable, indirect LSD radix sort for unsigned 32-bit keys. */
class RadixSort {
	private readonly outputAttribute: StorageBufferAttribute;
	private readonly device: GPUDevice | null;
	private readonly passResources: PassResources[];
	private readonly scanBlockHistogramsPipeline: GPUComputePipeline | null;
	private readonly scanBlockHistogramsBindGroup: GPUBindGroup | null;
	private readonly scanDigitTotalsPipeline: GPUComputePipeline | null;
	private readonly scanDigitTotalsBindGroup: GPUBindGroup | null;
	private readonly dispatchBuffer: GPUBuffer | null;
	private readonly ownedBuffers: GPUBuffer[];

	constructor(
		renderer: WebGPURenderer,
		keys: StorageBufferAttribute,
		indices: StorageBufferAttribute,
		output: StorageBufferAttribute,
		options: RadixSortOptions = {}
	) {
		const length = options.length ?? indices.count;
		if (length > indices.count) throw new RangeError('length exceeds input index capacity');
		if (length > output.count) throw new RangeError('length exceeds output index capacity');
		if (keys.itemSize !== 1 || indices.itemSize !== 1 || output.itemSize !== 1)
			throw new TypeError('keys, indices, and output must have itemSize 1');
		if (indices === output) throw new TypeError('input and output index attributes must be different');

		this.outputAttribute = output;
		this.device = null;
		this.passResources = [];
		this.scanBlockHistogramsPipeline = null;
		this.scanBlockHistogramsBindGroup = null;
		this.scanDigitTotalsPipeline = null;
		this.scanDigitTotalsBindGroup = null;
		this.dispatchBuffer = null;
		this.ownedBuffers = [];

		// Avoid creating zero-sized WebGPU buffers. sort() is a no-op for this case.
		if (length === 0) return;

		const backend = renderer.backend as unknown as WebGPUBackendAccess;
		if (backend.isWebGPUBackend !== true || backend.device === null) {
			throw new Error('RadixSort requires an initialized WebGPURenderer with a WebGPU backend');
		}

		const device = backend.device;
		this.device = device;

		const getStorageBuffer = (attribute: StorageBufferAttribute): GPUBuffer => {
			backend.createStorageAttribute(attribute);
			const buffer = backend.get(attribute).buffer;
			if (buffer === undefined) throw new Error('Three.js did not create the storage buffer');
			return buffer;
		};

		const keyBuffer = getStorageBuffer(keys);
		const indexBuffer = getStorageBuffer(indices);
		const outputBuffer = getStorageBuffer(output);
		const scratchBuffer = this.createBuffer(indices.count * Uint32Array.BYTES_PER_ELEMENT);

		const blockCount = Math.ceil(length / BLOCK_ITEMS);
		const histogramBytes = blockCount * RADIX_SIZE * Uint32Array.BYTES_PER_ELEMENT;
		const blockHistograms = this.createBuffer(histogramBytes);
		const blockPrefixes = this.createBuffer(histogramBytes);
		const digitTotals = this.createBuffer(RADIX_SIZE * Uint32Array.BYTES_PER_ELEMENT);
		const digitOffsets = this.createBuffer(RADIX_SIZE * Uint32Array.BYTES_PER_ELEMENT);

		const dispatchBuffer = device.createBuffer({
			label: 'radix-dispatch-arguments',
			size: 4 * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
			mappedAtCreation: true
		});
		new Uint32Array(dispatchBuffer.getMappedRange()).set([Math.ceil(blockCount / WORKGROUP_SIZE), 1, 1, length]);
		dispatchBuffer.unmap();
		this.dispatchBuffer = dispatchBuffer;
		this.ownedBuffers.push(dispatchBuffer);

		const histogramModule = device.createShaderModule({ code: histogramShader });
		const scatterModule = device.createShaderModule({ code: scatterShader });
		const scanBlockHistogramsModule = device.createShaderModule({
			code: scanBlockHistogramsShader
		});
		const scanDigitTotalsModule = device.createShaderModule({ code: scanDigitTotalsShader });

		const scanBlockHistogramsPipeline = device.createComputePipeline({
			layout: 'auto',
			compute: { module: scanBlockHistogramsModule }
		});
		this.scanBlockHistogramsPipeline = scanBlockHistogramsPipeline;
		this.scanBlockHistogramsBindGroup = device.createBindGroup({
			layout: scanBlockHistogramsPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: blockHistograms } },
				{ binding: 1, resource: { buffer: blockPrefixes } },
				{ binding: 2, resource: { buffer: digitTotals } },
				{ binding: 3, resource: { buffer: dispatchBuffer } }
			]
		});

		const scanDigitTotalsPipeline = device.createComputePipeline({
			layout: 'auto',
			compute: { module: scanDigitTotalsModule }
		});
		this.scanDigitTotalsPipeline = scanDigitTotalsPipeline;
		this.scanDigitTotalsBindGroup = device.createBindGroup({
			layout: scanDigitTotalsPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: digitTotals } },
				{ binding: 1, resource: { buffer: digitOffsets } }
			]
		});

		const buffers = [indexBuffer, scratchBuffer, outputBuffer];
		for (let pass = 0; pass < PASSES; pass++) {
			const source = pass === 0 ? 0 : pass % 2 === 1 ? 1 : 2;
			const destination = pass % 2 === 0 ? 1 : 2;
			const shift = pass * RADIX_BITS;

			const histogramPipeline = device.createComputePipeline({
				layout: 'auto',
				compute: { module: histogramModule, constants: { shift } }
			});
			const histogramBindGroup = device.createBindGroup({
				layout: histogramPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: keyBuffer } },
					{ binding: 1, resource: { buffer: buffers[source] } },
					{ binding: 2, resource: { buffer: blockHistograms } },
					{ binding: 3, resource: { buffer: dispatchBuffer } }
				]
			});

			const scatterPipeline = device.createComputePipeline({
				layout: 'auto',
				compute: { module: scatterModule, constants: { shift } }
			});
			const scatterBindGroup = device.createBindGroup({
				layout: scatterPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: keyBuffer } },
					{ binding: 1, resource: { buffer: buffers[source] } },
					{ binding: 2, resource: { buffer: blockPrefixes } },
					{ binding: 3, resource: { buffer: digitOffsets } },
					{ binding: 4, resource: { buffer: buffers[destination] } },
					{ binding: 5, resource: { buffer: dispatchBuffer } }
				]
			});

			this.passResources.push({
				histogramPipeline,
				histogramBindGroup,
				scatterPipeline,
				scatterBindGroup
			});
		}
	}

	private createBuffer(size: number): GPUBuffer {
		if (this.device === null) throw new Error('Cannot create a buffer without a GPU device');
		const buffer = this.device.createBuffer({
			size,
			usage: GPUBufferUsage.STORAGE
		});
		this.ownedBuffers.push(buffer);
		return buffer;
	}

	sort(): StorageBufferAttribute {
		if (this.device === null || this.dispatchBuffer === null) return this.outputAttribute;

		const encoder = this.device.createCommandEncoder({ label: 'radix-sort' });
		const computePass = encoder.beginComputePass({ label: 'radix-sort' });

		for (const resources of this.passResources) {
			computePass.setPipeline(resources.histogramPipeline);
			computePass.setBindGroup(0, resources.histogramBindGroup);
			computePass.dispatchWorkgroupsIndirect(this.dispatchBuffer, 0);

			computePass.setPipeline(this.scanBlockHistogramsPipeline!);
			computePass.setBindGroup(0, this.scanBlockHistogramsBindGroup!);
			computePass.dispatchWorkgroups(1);

			computePass.setPipeline(this.scanDigitTotalsPipeline!);
			computePass.setBindGroup(0, this.scanDigitTotalsBindGroup!);
			computePass.dispatchWorkgroups(1);

			computePass.setPipeline(resources.scatterPipeline);
			computePass.setBindGroup(0, resources.scatterBindGroup);
			computePass.dispatchWorkgroupsIndirect(this.dispatchBuffer, 0);
		}

		computePass.end();
		this.device.queue.submit([encoder.finish()]);
		return this.outputAttribute;
	}

	async sortAsync(): Promise<StorageBufferAttribute> {
		const output = this.sort();
		if (this.device !== null) await this.device.queue.onSubmittedWorkDone();
		return output;
	}

	dispose(): void {
		for (const buffer of this.ownedBuffers) buffer.destroy();
	}
}

export { RadixSort };
export const RadixSortConfig = Object.freeze({
	RADIX_BITS,
	RADIX_SIZE,
	PASSES,
	BLOCK_ITEMS,
	WORKGROUP_SIZE
});
