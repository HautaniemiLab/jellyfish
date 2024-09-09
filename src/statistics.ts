export function klDivergence(P: number[], Q: number[]): number {
  let klDiv = 0;
  for (let i = 0; i < P.length; i++) {
    if (P[i] !== 0 && Q[i] !== 0) {
      klDiv += P[i] * Math.log(P[i] / Q[i]);
    }
  }
  return klDiv;
}

export function jsDivergence(P: number[], Q: number[]): number {
  const M = P.map((p, i) => (p + Q[i]) / 2);
  return 0.5 * klDivergence(P, M) + 0.5 * klDivergence(Q, M);
}

/**
 *
 * @param data Input data
 * @param distanceFunction Function to calculate distance or divergence between two data points
 */
export function createDistanceMatrix<T>(
  data: T[],
  distanceFunction: (P: T, Q: T) => number
) {
  const n = data.length;
  const matrix = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const distance = distanceFunction(data[i], data[j]);
      matrix[i][j] = distance;
      matrix[j][i] = distance;
    }
  }
  return matrix;
}
