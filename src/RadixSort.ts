import { StorageBufferAttribute } from 'three';
import type WebGPURenderer from 'three/addons/renderers/webgpu/WebGPURenderer.js';
import {
	Fn,
	If,
	atomicAdd,
	globalId,
	localId,
	storage,
	uint,
	uniform,
	workgroupArray,
	workgroupBarrier,
	workgroupId
} from 'three/tsl';

const RADIX_BITS = 4;
const RADIX_SIZE = 1 << RADIX_BITS;
const PASSES = 32 / RADIX_BITS;
const WORKGROUP_SIZE = 256;

export interface RadixSortOptions {
	/** Number of entries at the beginning of the index buffer to sort. */
	length?: number;
}

/**
 * Stable, indirect LSD radix sort for unsigned 32-bit keys.
 *
 * The immutable key buffer is addressed through the index buffer. Only the
 * indices are ping-ponged, so duplicate keys retain their input order.
 */
class RadixSort {
	private readonly renderer: WebGPURenderer;
	private readonly length: number;
	private readonly maxElements: number;
	private readonly maxGroups: number;
	private readonly groupCount: number;
	private readonly keysAttribute: StorageBufferAttribute;
	private readonly inputAttribute: StorageBufferAttribute;
	private readonly outputAttribute: StorageBufferAttribute;
	private readonly scratchAttribute: StorageBufferAttribute;
	private readonly keys: any;
	private readonly buffers: any[];
	private readonly histograms: any;
	private readonly offsets: any;
	private readonly shift: any;
	private readonly count: any;
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
		this.maxElements = indices.count;

		if (this.length > this.maxElements) throw new RangeError('length exceeds input index capacity');
		if (this.length > output.count) throw new RangeError('length exceeds output index capacity');
		if (keys.itemSize !== 1 || indices.itemSize !== 1 || output.itemSize !== 1)
			throw new TypeError('keys, indices, and output must have itemSize 1');
		if (indices === output) throw new TypeError('input and output index attributes must be different');

		this.maxGroups = Math.max(1, Math.ceil(this.maxElements / WORKGROUP_SIZE));
		this.groupCount = Math.max(1, Math.ceil(this.length / WORKGROUP_SIZE));
		this.keysAttribute = keys;
		this.inputAttribute = indices;
		this.outputAttribute = output;
		this.scratchAttribute = new StorageBufferAttribute(new Uint32Array(this.maxElements), 1);

		this.keys = storage(keys, 'uint', keys.count).toReadOnly();
		this.buffers = [
			storage(this.inputAttribute, 'uint', this.maxElements),
			storage(this.scratchAttribute, 'uint', this.maxElements),
			storage(this.outputAttribute, 'uint', output.count)
		];
		this.histograms = storage(
			new StorageBufferAttribute(new Uint32Array(this.maxGroups * RADIX_SIZE), 1),
			'uint',
			this.maxGroups * RADIX_SIZE
		);
		this.offsets = storage(
			new StorageBufferAttribute(new Uint32Array(this.maxGroups * RADIX_SIZE), 1),
			'uint',
			this.maxGroups * RADIX_SIZE
		);
		this.shift = uniform(0, 'uint');
		this.count = uniform(this.length, 'uint');

		this._buildKernels();
	}

	_buildKernels() {
		const localHistogram = workgroupArray('uint', RADIX_SIZE);
		const tileDigits = workgroupArray('uint', WORKGROUP_SIZE);

		this.histogramNodes = this.buffers.map((input) =>
			Fn(() => {
				const lid = localId.x;
				If(lid.lessThan(RADIX_SIZE), () => localHistogram.element(lid).assign(0));
				workgroupBarrier();

				const position = globalId.x;
				If(position.lessThan(this.count), () => {
					const sourceIndex = input.element(position);
					const digit = this.keys
						.element(sourceIndex)
						.shiftRight(this.shift)
						.bitAnd(RADIX_SIZE - 1);
					atomicAdd(localHistogram.element(digit), 1);
				});
				workgroupBarrier();

				If(lid.lessThan(RADIX_SIZE), () => {
					const address = uint(workgroupId.x).mul(RADIX_SIZE).add(lid);
					this.histograms.element(address).assign(localHistogram.element(lid));
				});
			})().compute(this.groupCount * WORKGROUP_SIZE, [WORKGROUP_SIZE])
		);

		// One workgroup scans all groups for each digit, then thread zero scans
		// the digit totals. This intentionally simple baseline has no subgroup
		// dependency and supports any number of input workgroups.
		const binTotals = workgroupArray('uint', RADIX_SIZE);
		this.scanNode = Fn(() => {
			const bin = localId.x;
			const total = uint(0).toVar();
			If(bin.lessThan(RADIX_SIZE), () => {
				for (let group = 0; group < this.maxGroups; group++) {
					If(uint(group).lessThan(this.groupCount), () => {
						const address = uint(group * RADIX_SIZE).add(bin);
						this.offsets.element(address).assign(total);
						total.addAssign(this.histograms.element(address));
					});
				}
				binTotals.element(bin).assign(total);
			});
			workgroupBarrier();

			If(bin.equal(0), () => {
				const base = uint(0).toVar();
				for (let digit = 0; digit < RADIX_SIZE; digit++) {
					for (let group = 0; group < this.maxGroups; group++) {
						If(uint(group).lessThan(this.groupCount), () => {
							const address = uint(group * RADIX_SIZE + digit);
							this.offsets.element(address).addAssign(base);
						});
					}
					base.addAssign(binTotals.element(digit));
				}
			});
		})().compute(WORKGROUP_SIZE, [WORKGROUP_SIZE]);

		const scatter = (input: any, output: any) =>
			Fn(() => {
				const position = globalId.x;
				const valid = position.lessThan(this.count);
				const digit = uint(RADIX_SIZE).toVar();
				const sourceIndex = uint(0).toVar();

				If(valid, () => {
					sourceIndex.assign(input.element(position));
					digit.assign(
						this.keys
							.element(sourceIndex)
							.shiftRight(this.shift)
							.bitAnd(RADIX_SIZE - 1)
					);
				});
				tileDigits.element(localId.x).assign(digit);
				workgroupBarrier();

				If(valid, () => {
					const rank = uint(0).toVar();
					for (let previous = 0; previous < WORKGROUP_SIZE; previous++) {
						If(uint(previous).lessThan(localId.x).and(tileDigits.element(previous).equal(digit)), () =>
							rank.addAssign(1)
						);
					}
					const offsetAddress = uint(workgroupId.x).mul(RADIX_SIZE).add(digit);
					output.element(this.offsets.element(offsetAddress).add(rank)).assign(sourceIndex);
				});
			})().compute(this.groupCount * WORKGROUP_SIZE, [WORKGROUP_SIZE]);

		// Pass zero copies input -> scratch. Subsequent passes ping-pong between
		// scratch and the caller-owned output, making the eighth/final pass land
		// in output without ever replacing that attribute.
		this.scatterNodes = [
			scatter(this.buffers[0], this.buffers[1]),
			scatter(this.buffers[1], this.buffers[2]),
			scatter(this.buffers[2], this.buffers[1])
		];
	}

	/**
	 * Enqueue all compute passes and return immediately. Use this in a render
	 * loop; commands submitted afterwards can consume the caller-owned output.
	 */
	sort(): StorageBufferAttribute {
		if (this.length === 0) return this.outputAttribute;

		for (let pass = 0; pass < PASSES; pass++) {
			this.shift.value = pass * RADIX_BITS;
			const source = pass === 0 ? 0 : pass % 2 === 1 ? 1 : 2;
			const scatter = pass === 0 ? 0 : pass % 2 === 1 ? 1 : 2;
			this.renderer.compute(this.histogramNodes[source]);
			this.renderer.compute(this.scanNode);
			this.renderer.compute(this.scatterNodes[scatter]);
		}

		return this.outputAttribute;
	}

	/** Enqueue each pass and wait for GPU completion before returning. */
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
