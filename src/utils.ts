export function smoothstep(edge0: number, edge1: number, x: number) {
  x = clamp(0, 1, (x - edge0) / (edge1 - edge0));
  return x * x * (3 - 2 * x);
}

export function smootherstep(edge0: number, edge1: number, x: number) {
  x = clamp(0, 1, (x - edge0) / (edge1 - edge0));
  return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0);
}

export function fancystep(
  edge0: number,
  edge1: number,
  x: number,
  tipShape = 0.1
) {
  const span = edge1 - edge0;
  const step = (x) =>
    smootherstep(edge0 - span * (1 / (1 - tipShape) - 1), edge1, x);
  const atZero = step(edge0);
  return Math.max(0, step(x) - atZero) / (1 - atZero);
}

export const lerp = (a: number, b: number, x: number) => (1 - x) * a + x * b;

export const clamp = (lower: number, upper: number, x: number) =>
  Math.max(lower, Math.min(upper, x));

export function SeededRNG(seed: number) {
  // From ChatGPT
  return function () {
    // Constants for a Linear Congruential Generator
    const a = 1664525;
    const c = 1013904223;
    const m = 4294967296; // 2^32
    // Update the seed and return a random value between 0 and 1
    seed = (a * seed + c) % m;
    return seed / m;
  };
}

export function fisherYatesShuffle<T>(array: T[], rng: () => number): T[] {
  // Adapted from ChatGPT

  // Make a copy to keep the function pure
  array = array.slice();

  let currentIndex = array.length,
    tmp,
    randomIndex;

  // While there remain elements to shuffle
  while (currentIndex !== 0) {
    // Pick a remaining element
    randomIndex = Math.floor(rng() * currentIndex);
    currentIndex -= 1;

    tmp = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = tmp;
  }

  return array;
}
