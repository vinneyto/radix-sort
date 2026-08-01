import { StorageBufferAttribute, type WebGPURenderer } from 'three/webgpu';
import { Fn, globalId, localId, storage, uniform, wgslFn, workgroupArray, workgroupId } from 'three/tsl';

const RADIX_BITS = 4;
const RADIX_SIZE = 1 << RADIX_BITS;
const PASSES = 32 / RADIX_BITS;
const WORKGROUP_SIZE = 256;

const histogramWGSL = wgslFn(/* wgsl */ `
	fn radix_histogram(
		keys: ptr<storage, array<u32>, read>,
		input: ptr<storage, array<u32>, read_write>,
		histograms: ptr<storage, array<u32>, read_write>,
		local_histogram: ptr<workgroup, array<atomic<u32>, ${RADIX_SIZE}>, read_write>,
		position: u32,
		lid: u32,
		group: u32,
		count: u32,
		shift: u32
	) -> void {
		if (lid < ${RADIX_SIZE}u) {
			atomicStore(&(*local_histogram)[lid], 0u);
		}
		workgroupBarrier();

		if (position < count) {
			let source_index = (*input)[position];
			let digit = ((*keys)[source_index] >> shift) & ${RADIX_SIZE - 1}u;
			atomicAdd(&(*local_histogram)[digit], 1u);
		}
		workgroupBarrier();

		if (lid < ${RADIX_SIZE}u) {
			(*histograms)[group * ${RADIX_SIZE}u + lid] = atomicLoad(&(*local_histogram)[lid]);
		}
	}
`);

const scanWGSL = wgslFn(/* wgsl */ `
	fn radix_scan(
		histograms: ptr<storage, array<u32>, read_write>,
		offsets: ptr<storage, array<u32>, read_write>,
		bin_totals: ptr<workgroup, array<u32, ${RADIX_SIZE}>, read_write>,
		bin: u32,
		group_count: u32
	) -> void {
		if (bin < ${RADIX_SIZE}u) {
			var total = 0u;
			for (var group = 0u; group < group_count; group++) {
				let address = group * ${RADIX_SIZE}u + bin;
				(*offsets)[address] = total;
				total += (*histograms)[address];
			}
			(*bin_totals)[bin] = total;
		}
		workgroupBarrier();

		if (bin == 0u) {
			var base = 0u;
			for (var digit = 0u; digit < ${RADIX_SIZE}u; digit++) {
				for (var group = 0u; group < group_count; group++) {
					(*offsets)[group * ${RADIX_SIZE}u + digit] += base;
				}
				base += (*bin_totals)[digit];
			}
		}
	}
`);

const scatterWGSL = wgslFn(/* wgsl */ `
	fn radix_scatter(
		keys: ptr<storage, array<u32>, read>,
		input: ptr<storage, array<u32>, read_write>,
		output: ptr<storage, array<u32>, read_write>,
		offsets: ptr<storage, array<u32>, read_write>,
		tile_digits: ptr<workgroup, array<u32, ${WORKGROUP_SIZE}>, read_write>,
		position: u32,
		lid: u32,
		group: u32,
		count: u32,
		shift: u32
	) -> void {
		let valid = position < count;
		var source_index = 0u;
		var digit = ${RADIX_SIZE}u;
		if (valid) {
			source_index = (*input)[position];
			digit = ((*keys)[source_index] >> shift) & ${RADIX_SIZE - 1}u;
		}
		(*tile_digits)[lid] = digit;
		workgroupBarrier();

		if (valid) {
			var rank = 0u;
			for (var previous = 0u; previous < lid; previous++) {
				if ((*tile_digits)[previous] == digit) {
					rank++;
				}
			}
			let offset_address = group * ${RADIX_SIZE}u + digit;
			(*output)[(*offsets)[offset_address] + rank] = source_index;
		}
	}
`);

export interface RadixSortOptions {
	/** Number of entries at the beginning of the index buffer to sort. */
	length?: number;
}

/** Stable, indirect LSD radix sort for unsigned 32-bit keys. */
class RadixSort {
	private readonly renderer: WebGPURenderer;
	private readonly length: number;
	private readonly outputAttribute: StorageBufferAttribute;
	private readonly shift: any;
	private histogramNodes!: any[];
	private scanNode!: any;
	private scatterNodes!: any[];

	constructor(
		renderer: WebGPURenderer,
		keys: StorageBufferAttribute,
		indices: StorageBufferAttribute,
		output: StorageBufferAttribute,
		options: RadixSortOptions = {}
	) {
		this.renderer = renderer;
		this.length = options.length ?? indices.count;
		if (this.length > indices.count) throw new RangeError('length exceeds input index capacity');
		if (this.length > output.count) throw new RangeError('length exceeds output index capacity');
		if (keys.itemSize !== 1 || indices.itemSize !== 1 || output.itemSize !== 1)
			throw new TypeError('keys, indices, and output must have itemSize 1');
		if (indices === output) throw new TypeError('input and output index attributes must be different');

		const maxGroups = Math.max(1, Math.ceil(indices.count / WORKGROUP_SIZE));
		const groupCount = Math.max(1, Math.ceil(this.length / WORKGROUP_SIZE));
		const keyBuffer = storage(keys, 'uint', keys.count).toReadOnly();
		const buffers = [
			storage(indices, 'uint', indices.count),
			storage(new StorageBufferAttribute(new Uint32Array(indices.count), 1), 'uint', indices.count),
			storage(output, 'uint', output.count)
		];
		const histograms = storage(
			new StorageBufferAttribute(new Uint32Array(maxGroups * RADIX_SIZE), 1),
			'uint',
			maxGroups * RADIX_SIZE
		);
		const offsets = storage(
			new StorageBufferAttribute(new Uint32Array(maxGroups * RADIX_SIZE), 1),
			'uint',
			maxGroups * RADIX_SIZE
		);
		this.outputAttribute = output;
		this.shift = uniform(0, 'uint');
		const count = uniform(this.length, 'uint');
		const groups = uniform(groupCount, 'uint');

		const localHistogram = workgroupArray(`atomic<u32>`, RADIX_SIZE);
		this.histogramNodes = buffers.map((input) =>
			Fn(() => {
				histogramWGSL({
					keys: keyBuffer,
					input,
					histograms,
					local_histogram: localHistogram,
					position: globalId.x,
					lid: localId.x,
					group: workgroupId.x,
					count,
					shift: this.shift
				});
			})().compute(groupCount * WORKGROUP_SIZE, [WORKGROUP_SIZE])
		);

		const binTotals = workgroupArray('uint', RADIX_SIZE);
		this.scanNode = Fn(() => {
			scanWGSL({ histograms, offsets, bin_totals: binTotals, bin: localId.x, group_count: groups });
		})().compute(WORKGROUP_SIZE, [WORKGROUP_SIZE]);

		const scatter = (input: any, destination: any) => {
			const tileDigits = workgroupArray('uint', WORKGROUP_SIZE);
			return Fn(() => {
				scatterWGSL({
					keys: keyBuffer,
					input,
					output: destination,
					offsets,
					tile_digits: tileDigits,
					position: globalId.x,
					lid: localId.x,
					group: workgroupId.x,
					count,
					shift: this.shift
				});
			})().compute(groupCount * WORKGROUP_SIZE, [WORKGROUP_SIZE]);
		};
		this.scatterNodes = [
			scatter(buffers[0], buffers[1]),
			scatter(buffers[1], buffers[2]),
			scatter(buffers[2], buffers[1])
		];
	}

	private enqueue(compute: (node: any) => unknown): void {
		for (let pass = 0; pass < PASSES; pass++) {
			this.shift.value = pass * RADIX_BITS;
			const source = pass === 0 ? 0 : pass % 2 === 1 ? 1 : 2;
			const scatter = pass === 0 ? 0 : pass % 2 === 1 ? 1 : 2;
			compute(this.histogramNodes[source]);
			compute(this.scanNode);
			compute(this.scatterNodes[scatter]);
		}
	}

	sort(): StorageBufferAttribute {
		if (this.length !== 0) this.enqueue((node) => this.renderer.compute(node));
		return this.outputAttribute;
	}

	async sortAsync(): Promise<StorageBufferAttribute> {
		if (this.length === 0) return this.outputAttribute;
		for (let pass = 0; pass < PASSES; pass++) {
			this.shift.value = pass * RADIX_BITS;
			const source = pass === 0 ? 0 : pass % 2 === 1 ? 1 : 2;
			const scatter = pass === 0 ? 0 : pass % 2 === 1 ? 1 : 2;
			await this.renderer.computeAsync(this.histogramNodes[source]);
			await this.renderer.computeAsync(this.scanNode);
			await this.renderer.computeAsync(this.scatterNodes[scatter]);
		}
		return this.outputAttribute;
	}
}

export { RadixSort };
export const RadixSortConfig = Object.freeze({ RADIX_BITS, RADIX_SIZE, PASSES, WORKGROUP_SIZE });
