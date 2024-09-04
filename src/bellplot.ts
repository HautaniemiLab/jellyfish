import { G } from "@svgdotjs/svg.js";
import { PhylogenyRow, Subclone } from "./data.js";
import { stratify, TreeNode } from "./tree.js";
import { clamp, lerp } from "./utils.js";
import * as d3 from "d3";

export interface BellPlotNode extends TreeNode<BellPlotNode> {
  subclone: Subclone;
  parentId: Subclone; // TODO: Remove this
  fraction: number;
  totalFraction: number;
  initialSize: number;
}

export interface BellPlotProperties {
  bellTipShape: number;
  bellTipSpread: number;
}

export function createBellPlotTree(
  phylogenyTable: PhylogenyRow[],
  proportionsMap: Map<Subclone, number>,
  preEmerged: Subclone[]
) {
  // TODO: Perhaps a single phylogenetic tree should be shared between all samples
  const root = stratify(
    phylogenyTable,
    (d) => d.subclone,
    (d) => d.parent,
    (d) =>
      ({
        subclone: d.subclone,
        parent: null,
        // Use the lookup table to join the proportions to the phylogeny
        fraction: proportionsMap.get(d.subclone) ?? 0,
        totalFraction: 0,
        children: [],
        // Initial size is zero if the subclone emerges in this sample.
        initialSize: preEmerged.includes(d.subclone) ? 1.0 : 0.0,
      } as BellPlotNode)
  );

  /**
   * For each node, calculate the sum of its fraction and all its descendant's fractions.
   * Actually, this is the same as cluster size.
   */
  function calculateTotalFractions(node: BellPlotNode) {
    let sum = node.fraction;
    for (const child of node.children) {
      sum += calculateTotalFractions(child);
    }
    node.totalFraction = sum;
    return sum;
  }

  /**
   * Normalize fractions so that the root node is 100% and all descendants
   * are scaled accordingly.
   */
  function normalizeChildren(node: BellPlotNode) {
    for (const child of node.children) {
      normalizeChildren(child);
    }

    const parent = node.parent;
    node.fraction =
      // Avoid NaNs (0 / 0 = NaN)
      node.totalFraction == 0
        ? 0
        : parent
        ? node.totalFraction / parent.totalFraction
        : 1;
  }

  calculateTotalFractions(root);
  normalizeChildren(root);

  return root;
}

/**
 * Adds the nested subclones into an SVG group.
 */
export function createBellPlotGroup(
  tree: BellPlotNode,
  shapers: Map<Subclone, Shaper>,
  subcloneColors: Map<Subclone, string>,
  width = 1,
  height = 1
) {
  const g = new G(); // SVG group

  /**
   * Draw a rectangle that is shaped using the shaper function.
   */
  function drawNode(node: BellPlotNode) {
    const shaper = shapers.get(node.subclone);

    // Skip zero-sized subclones. Otherwise they would be drawn as just a stroke, which is useless.
    if (node.totalFraction < 0.001) {
      return;
    }

    const upper1 = shaper(0, 0),
      upper2 = shaper(1, 0),
      lower1 = shaper(0, 1),
      lower2 = shaper(1, 1);

    let element;

    // Round to one decimal place to make the SVG smaller
    const r = (x: number) => Math.round(x * 10) / 10;

    if (upper1 == upper2 && lower1 == lower2) {
      // Seems to be a rectangle.
      element = g
        .rect(r(width), r((lower1 - upper1) * height))
        .move(0, r(upper1 * height));
    } else {
      // Not a rectangle. Let's draw a bell.

      // Segment count. Higher number produces smoother curves.
      const sc = 50;

      // Find the first segment where the subclone starts to emerge
      let firstSegment = 0;
      for (let i = 0; i <= sc; i++) {
        const x = i / sc;

        if (shaper(x, 0) - shaper(x, 1) != 0) {
          // The one where upper and lower edges collide
          firstSegment = Math.max(0, i - 1);
          break;
        }
      }

      const p = d3.pathRound(1);
      // Start the path
      p.moveTo(
        (firstSegment / sc) * width,
        shaper(firstSegment / sc, 1) * height
      );

      // Upper part of the bell path
      for (let i = firstSegment + 1; i <= sc; i++) {
        const x = i / sc;
        p.lineTo(x * width, shaper(x, 1) * height);
      }
      // Lower part of the bell path
      for (let i = sc; i >= firstSegment; i--) {
        const x = i / sc;
        p.lineTo(x * width, shaper(x, 0) * height);
      }

      // Close the path
      p.lineTo(
        (firstSegment / sc) * width,
        shaper(firstSegment / sc, 1) * height
      );

      element = g.path(p.toString());
    }

    const color = subcloneColors.get(node.subclone);
    element
      .fill(color)
      .stroke(d3.color(color).darker(0.6))
      .addClass("subclone");

    for (const child of node.children) {
      drawNode(child);
    }
  }

  drawNode(tree);

  return g;
}

export type Shaper = (x: number, y: number) => number;

/**
 * Creates shaper functions for each subclone. The shapers are nested, which
 * ensures that descendants always stay within the boundaries of their parents.
 */
export function treeToShapers(
  tree: BellPlotNode,
  props: BellPlotProperties
): Map<Subclone, Shaper> {
  const shapers: Map<Subclone, Shaper> = new Map();

  function process(
    node: BellPlotNode,
    parentShaper: Shaper = (x, y) => y,
    fractionalDepth = 0
  ) {
    const parentNode = node.parent;
    const remainingDepth = getDepth(node);

    const fractionalStep =
      node.initialSize == 0 ? (1 - fractionalDepth) / (remainingDepth + 1) : 0;

    if (parentNode) {
      fractionalDepth += fractionalStep;
    } else {
      // Root node has a special handling: no step is added
    }

    const fancy = (x: number) => {
      const tipShape = clamp(0, 0.9999, props.bellTipShape);
      return fancystep(fractionalDepth, 1, x, tipShape);
    };

    const shaper: Shaper = (x, y) =>
      parentShaper(
        x,
        // The fractionalChildDepth defines when the bell starts to appear.
        lerp(fancy(x), 1, node.initialSize) * node.fraction * (y - 0.5) + 0.5
      );

    shapers.set(node.subclone, shaper);

    // Children emerge as spread to better emphasize what their parent is
    const spreadPositions = stackChildren(node, true);
    // They end up as stacked to make the perception of the proportions easier
    const stackedPositions = stackChildren(node, false);

    const makeInterpolateSpreadStacked = (childIdx: number) => {
      if (node.initialSize == 0) {
        // Make an interpolator that smoothly interpolates between the spread and stacked positions
        return (x: number) => {
          const currentDepth = fractionalDepth + fractionalStep;
          let a = currentDepth >= 1 ? 1 : smoothstep(currentDepth, 1, x);
          const s = 1 - props.bellTipSpread;
          a = a * (1 - s) + s;
          return lerp(spreadPositions[childIdx], stackedPositions[childIdx], a);
        };
      } else {
        // Spread positions make no sense when the parent has an initialSize greater than zero
        return () => stackedPositions[childIdx];
      }
    };

    for (let i = 0; i < node.children.length; i++) {
      const childNode = node.children[i];

      const interpolateSpreadStacked = makeInterpolateSpreadStacked(i);

      process(
        childNode,
        (x, y) => shaper(x, y + interpolateSpreadStacked(x)),
        fractionalDepth
      );
    }
  }

  process(tree);

  return shapers;
}

/**
 * Returns the position of the children's top edges.
 */
function stackChildren(node: BellPlotNode, spread = false) {
  // Fractions wrt. the parent
  const fractions = node.children.map((n) => n.fraction);
  const positions = [];

  const remainingSpace = 1 - fractions.reduce((a, c) => a + c, 0);

  // Stack or spread?
  const spacing = spread ? remainingSpace / (fractions.length + 1) : 0;
  let cumSum = spread ? spacing : remainingSpace;

  for (const x of fractions) {
    positions.push(cumSum + (x - 1) / 2);
    cumSum += x + spacing;
  }

  return positions;
}

/**
 * Takes the shapers and builds stacked regions of the subclone
 * proportions visible at the left or right edge of the bells.
 * Returns a Map that maps node ids to an extents.
 *
 * This has two use cases:
 * 1. Render a pretty stacked bar chart
 * 2. Get attachment areas for the tentacles
 *
 * edge: 0 = left, 1 = right
 */
export function calculateSubcloneRegions(
  tree: BellPlotNode,
  shapers: Map<string, Shaper>,
  edge: 0 | 1 = 1
) {
  const regions = new Map<Subclone, [number, number]>();

  function process(node: BellPlotNode): number {
    const nodeShaper = shapers.get(node.subclone);
    const top = nodeShaper(edge, 0);

    const nodeBottom = nodeShaper(edge, 1);
    let bottom = nodeBottom;
    for (const child of node.children) {
      const childBottom = process(child);
      if (child.totalFraction > 0) {
        bottom = Math.min(bottom, childBottom);
      }
    }
    regions.set(node.subclone, [top, bottom]);

    return top;
  }

  process(tree);

  // Left edge needs some post processing because the tips of the
  // nested bells are not located at the bottom of their parents.
  const regionArray = Array.from(regions.values())
    .filter((v) => v[1] - v[0] > 0)
    .sort((a, b) => a[0] - b[0]);
  for (let i = 0; i < regionArray.length - 1; i++) {
    regionArray[i][1] = Math.max(regionArray[i][1], regionArray[i + 1][0]);
  }
  if (regionArray.length > 0) {
    // Extra special handling for the most bottom extent.
    regionArray[regionArray.length - 1][1] = Array.from(shapers.values())
      .map((shaper) => shaper(edge, 1))
      .filter((bottom) => !isNaN(bottom))
      .reduce((a, b) => Math.max(a, b), 0);
  }

  return regions;
}

function getDepth(node: BellPlotNode): number {
  return (
    (node.children ?? [])
      .filter((n) => n.fraction > 0)
      .map((n) => getDepth(n))
      .reduce((a, b) => Math.max(a, b), 0) + 1
  );
}

function smoothstep(edge0: number, edge1: number, x: number) {
  x = clamp(0, 1, (x - edge0) / (edge1 - edge0));
  return x * x * (3 - 2 * x);
}

function smootherstep(edge0: number, edge1: number, x: number) {
  x = clamp(0, 1, (x - edge0) / (edge1 - edge0));
  return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0);
}

function fancystep(edge0: number, edge1: number, x: number, tipShape = 0.1) {
  const span = edge1 - edge0;
  const step = (x: number) =>
    smootherstep(edge0 - span * (1 / (1 - tipShape) - 1), edge1, x);
  const atZero = step(edge0);
  return Math.max(0, step(x) - atZero) / (1 - atZero);
}
