export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Returns a bounding box for a collection of rectangles.
 */
export function getBoundingBox(rects: Iterable<Rect>) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const rect of rects) {
    minX = Math.min(minX, rect.x);
    minY = Math.min(minY, rect.y);
    maxX = Math.max(maxX, rect.x + rect.width);
    maxY = Math.max(maxY, rect.y + rect.height);
  }

  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}
