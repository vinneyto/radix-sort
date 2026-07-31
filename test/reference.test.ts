import { describe, expect, test } from 'vitest';
import { referenceSort, validateSort } from '../src/reference';

describe( 'Array.sort reference', () => {

	test( 'is indirect and stable', () => {

		const keys = Uint32Array.of( 5, 2, 5, 5, 2 );
		const indices = Uint32Array.of( 3, 0, 4, 2, 1 );
		expect( referenceSort( keys, indices ) ).toEqual( [ 4, 1, 3, 0, 2 ] );
		expect( validateSort( keys, indices, Uint32Array.of( 4, 1, 3, 0, 2 ) ) ).toBe( true );

	} );

	test( 'handles empty and repeated indices', () => {

		expect( referenceSort( Uint32Array.of(), Uint32Array.of() ) ).toEqual( [] );
		expect( referenceSort( Uint32Array.of( 9, 1 ), Uint32Array.of( 0, 1, 1, 0 ) ) ).toEqual( [ 1, 1, 0, 0 ] );

	} );

	test( 'validation rejects a wrong permutation', () => {

		expect( () => validateSort( [ 2, 1 ], [ 0, 1 ], [ 0, 1 ] ) ).toThrow( /differs/ );

	} );

} );
