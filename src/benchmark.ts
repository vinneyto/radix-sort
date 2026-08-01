/** Return how many times faster the candidate is than the baseline. */
export function speedupRatio(baselineMilliseconds: number, candidateMilliseconds: number): number {
	return baselineMilliseconds / candidateMilliseconds;
}
