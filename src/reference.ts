export type UnsignedKeys = ArrayLike<number>;
export type Indices = ArrayLike<number> & Iterable<number>;

export function referenceSort( keys: UnsignedKeys, indices: Indices ): number[] {

	return Array.from( indices, ( index, position ) => ( { index, position } ) )
		.sort( ( a, b ) => keys[ a.index ] - keys[ b.index ] || a.position - b.position )
		.map( entry => entry.index );

}

export function validateSort( keys: UnsignedKeys, input: Indices, actual: Indices ): true {

	const expected = referenceSort( keys, input );
	if ( expected.length !== actual.length || expected.some( ( value, i ) => value !== actual[ i ] ) ) {

		throw new Error( `GPU result differs from Array.sort reference\nexpected: ${ expected }\nactual:   ${ Array.from( actual ) }` );

	}
	return true;

}
