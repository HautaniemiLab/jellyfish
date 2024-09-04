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
  // Make a copy to keep the function pure
  array = array.slice();

  // While there remain elements to shuffle
  for (let i = array.length - 1; i > 0; i--) {
    const randomIndex = Math.floor(rng() * (i + 1));

    const tmp = array[i];
    array[i] = array[randomIndex];
    array[randomIndex] = tmp;
  }

  return array;
}

export function mapUnion<K, V>(...maps: Array<Map<K, V>>): Map<K, V> {
  const union = new Map<K, V>();
  for (const m of maps) {
    for (const [k, v] of m) {
      union.set(k, v);
    }
  }
  return union;
}
