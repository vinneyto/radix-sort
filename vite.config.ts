import { defineConfig } from 'vitest/config';

export default defineConfig( {
	server: {
		open: true
	},
	test: {
		include: [ 'test/**/*.test.ts' ]
	}
} );
