import { G } from "@svgdotjs/svg.js";
import { Subclone } from "./data.js";
import { clamp, lerp } from "./utils.js";
import * as d3 from "d3";
import { PhylogenyNode } from "./phylogeny.js";
import { SubcloneMetricsMap } from "./composition.js";

export interface BellPlotProperties {
  bellTipShape: number;
  bellTipSpread: number;
}

/**
 * Adds the nested subclones into an SVG group.
 */
export function createBellPlotGroup(
  tree: PhylogenyNode,
  shapers: Map<Subclone, Shaper>,
  subcloneColors: Map<Subclone, string>,
  width = 1,
  height = 1
) {
  const g = new G(); // SVG group
  g.addClass("bell");

  /**
   * Draw a rectangle that is shaped using the shaper function.
   */
  function drawNode(node: PhylogenyNode) {
    const shaper = shapers.get(node.subclone);

    const upper1 = shaper(0, 0),
      upper2 = shaper(1, 0),
      lower1 = shaper(0, 1),
      lower2 = shaper(1, 1);

    // Skip zero-sized subclones. Otherwise they would be drawn as just a stroke, which is useless.
    if (Math.abs(upper2 - lower2) < 0.01) {
      return;
    }

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
  phylogenyRoot: PhylogenyNode,
  metricsMap: SubcloneMetricsMap,
  preEmergedSubclones: Set<Subclone>,
  props: BellPlotProperties
): Map<Subclone, Shaper> {
  function getDepth(node: PhylogenyNode): number {
    return (
      (node.children ?? [])
        .filter((n) => metricsMap.get(n.subclone).clusterSize > 0.001) // TODO: epsilon
        .map((n) => getDepth(n))
        .reduce((a, b) => Math.max(a, b), 0) + 1
    );
  }

  const shapers: Map<Subclone, Shaper> = new Map();

  function traverse(
    node: PhylogenyNode,
    parentShaper: Shaper = (x, y) => y,
    fractionalDepth = 0
  ) {
    const parentNode = node.parent;
    const subclone = node.subclone;
    const metrics = metricsMap.get(subclone);
    const preEmerged = preEmergedSubclones.has(subclone);

    const remainingDepth = getDepth(node);

    const fractionalStep = preEmerged
      ? 0
      : (1 - fractionalDepth) / (remainingDepth + 1);

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
        lerp(fancy(x), 1, preEmerged ? 1 : 0) *
          metrics.fractionOfParent *
          (y - 0.5) +
          0.5
      );

    shapers.set(node.subclone, shaper);

    // Children emerge as spread to better emphasize what their parent is
    const spreadPositions = stackChildren(node, metricsMap, true);
    // They end up as stacked to make the perception of the proportions easier
    const stackedPositions = stackChildren(node, metricsMap, false);

    const makeInterpolateSpreadStacked = (childIdx: number) => {
      if (preEmerged) {
        // Spread positions make no sense when the parent has an initialSize greater than zero
        return () => stackedPositions[childIdx];
      } else {
        // Make an interpolator that smoothly interpolates between the spread and stacked positions
        return (x: number) => {
          const currentDepth = fractionalDepth + fractionalStep;
          let a = currentDepth >= 1 ? 1 : smoothstep(currentDepth, 1, x);
          const s = 1 - props.bellTipSpread;
          a = a * (1 - s) + s;
          return lerp(spreadPositions[childIdx], stackedPositions[childIdx], a);
        };
      }
    };

    for (let i = 0; i < node.children.length; i++) {
      const childNode = node.children[i];

      const interpolateSpreadStacked = makeInterpolateSpreadStacked(i);

      traverse(
        childNode,
        (x, y) => shaper(x, y + interpolateSpreadStacked(x)),
        fractionalDepth
      );
    }
  }

  traverse(phylogenyRoot);

  return shapers;
}

/**
 * Returns the position of the children's top edges.
 */
function stackChildren(
  node: PhylogenyNode,
  metricsMap: SubcloneMetricsMap,
  spread = false
) {
  // Fractions wrt. the parent
  const fractions = node.children.map(
    (n) => metricsMap.get(n.subclone).fractionOfParent
  );
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
 * Returns a Map that maps node ids to regions.
 *
 * This has two use cases:
 * 1. Get attachment areas for the tentacles
 * 2. Render a pretty stacked bar chart
 *
 * edge: 0 = left, 1 = right
 */
export function calculateSubcloneRegions(
  phylogenyRoot: PhylogenyNode,
  metricsMap: SubcloneMetricsMap,
  shapers: Map<string, Shaper>,
  edge: 0 | 1 = 1
) {
  const regions = new Map<Subclone, [number, number]>();

  function process(node: PhylogenyNode): number {
    const nodeShaper = shapers.get(node.subclone);
    const top = nodeShaper(edge, 0);

    const nodeBottom = nodeShaper(edge, 1);
    let bottom = nodeBottom;
    for (const child of node.children) {
      const childBottom = process(child);
      if (metricsMap.get(child.subclone).clusterSize > 0) {
        bottom = Math.min(bottom, childBottom);
      }
    }
    regions.set(node.subclone, [top, bottom]);

    return top;
  }

  process(phylogenyRoot);

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
