export type UnsignedKeys = ArrayLike<number>;
export type Indices = ArrayLike<number> & Iterable<number>;

export function referenceSort(keys: UnsignedKeys, indices: Indices): number[] {
	// ECMAScript requires Array.sort to be stable. Comparing keys alone makes
	// this both the direct CPU baseline and the stability oracle for radix sort.
	return Array.from(indices).sort((a, b) => keys[a] - keys[b]);
}

export function validateSort(keys: UnsignedKeys, input: Indices, actual: Indices): true {
	const expected = referenceSort(keys, input);
	if (expected.length !== actual.length || expected.some((value, i) => value !== actual[i])) {
		throw new Error(
			`GPU result differs from Array.sort reference\nexpected: ${expected}\nactual:   ${Array.from(actual)}`
		);
	}
	return true;
}
