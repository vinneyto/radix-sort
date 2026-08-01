/** Return baseline time as a percentage of candidate time. */
export function relativeTimePercent(baselineMilliseconds: number, candidateMilliseconds: number): number {
	return (baselineMilliseconds / candidateMilliseconds) * 100;
}
