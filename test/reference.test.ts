import { describe, expect, test } from 'vitest';
import { radixSort as threeRadixSort } from 'three/addons/utils/SortUtils.js';
import { referenceSort, validateSort } from '../src/reference';
import { speedupRatio } from '../src/benchmark';

describe('Array.sort reference', () => {
	test('is indirect and stable', () => {
		const keys = Uint32Array.of(5, 2, 5, 5, 2);
		const indices = Uint32Array.of(3, 0, 4, 2, 1);
		expect(referenceSort(keys, indices)).toEqual([4, 1, 3, 0, 2]);
		expect(validateSort(keys, indices, Uint32Array.of(4, 1, 3, 0, 2))).toBe(true);
	});

	test('handles empty and repeated indices', () => {
		expect(referenceSort(Uint32Array.of(), Uint32Array.of())).toEqual([]);
		expect(referenceSort(Uint32Array.of(9, 1), Uint32Array.of(0, 1, 1, 0))).toEqual([1, 1, 0, 0]);
	});

	test('validation rejects a wrong permutation', () => {
		expect(() => validateSort([2, 1], [0, 1], [0, 1])).toThrow(/differs/);
	});
});

describe('benchmark ratios', () => {
	test('reports how many times faster the GPU candidate is than the CPU baseline', () => {
		expect(speedupRatio(12.12, 3.5)).toBeCloseTo(3.462857142857143);
		expect(speedupRatio(1, 2)).toBe(0.5);
	});
});

describe('Three.js SortUtils radixSort', () => {
	test('supports the same indirect stable ordering contract', () => {
		const keys = Uint32Array.of(5, 2, 5, 5, 2);
		const indices = [3, 0, 4, 2, 1];
		threeRadixSort(indices, { get: (index) => keys[index] });
		expect(indices).toEqual(referenceSort(keys, [3, 0, 4, 2, 1]));
	});
});
