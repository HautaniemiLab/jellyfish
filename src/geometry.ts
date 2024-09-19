export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Returns a bounding box for a collection of rectangles.
 */
export function getBoundingBox(rects: Iterable<Rect>): Rect {
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

export function getIntersection(a: Rect, b: Rect): Rect {
  const x = Math.max(a.x, b.x);
  const y = Math.max(a.y, b.y);
  const width = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - x);
  const height = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - y);

  if (width > 0 && height > 0) {
    return { x, y, width, height };
  } else {
    return null;
  }
}

export function isIntersecting(rect: Rect, rects: Iterable<Rect>): boolean {
  for (const other of rects) {
    if (getIntersection(rect, other) !== null) {
      return true;
    }
  }

  return false;
}

export function addPadding(rect: Rect, padding: number): Rect {
  return {
    x: rect.x - padding,
    y: rect.y - padding,
    width: rect.width + 2 * padding,
    height: rect.height + 2 * padding,
  };
}
