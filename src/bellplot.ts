import { G, SVG } from "@svgdotjs/svg.js";
import { Subclone } from "./data.js";
import { clamp, lerp } from "./utils.js";
import * as d3 from "d3";
import { PhylogenyNode } from "./phylogeny.js";
import { SubcloneMetrics, SubcloneMetricsMap } from "./composition.js";
import { treeToNodeArray } from "./tree.js";
import { drawArrowAndLabel } from "./utilityElements.js";

export interface BellPlotProperties {
  /**
   * The shape of the bell tip. 0 is a sharp tip, 1 is a blunt tip.
   *
   * @minimum 0
   * @maximum 1
   * @default 0.1
   */
  bellTipShape: number;

  /**
   * How much to spread nested bell tips. 0 is no spread, 1 is full spread.
   *
   * @minimum 0
   * @maximum 1
   * @default 0.5
   */
  bellTipSpread: number;

  /**
   * The width of strokes in the bell.
   *
   * @minimum 0
   * @maximum 10
   * @default 1
   */
  bellStrokeWidth: number;

  /**
   * How much the stroke color of the bells is darkened.
   *
   * @minimum 0
   * @maximum 2
   * @default 0.6
   */
  bellStrokeDarkening: number;

  /**
   * Where the bell has fully appeared and the plateau starts.
   *
   * @minimum 0
   * @maximum 1
   * @default 0.70
   */
  bellPlateauPos: number;
}

/**
 * Adds the nested subclones into an SVG group.
 */
export function drawBellPlot(
  container: G,
  tree: PhylogenyNode,
  shapers: Map<Subclone, Shaper>,
  passThroughSubclones: Set<Subclone>,
  subcloneColors: Map<Subclone, string>,
  bellPlotProperties: BellPlotProperties,
  width = 1,
  height = 1,
  passThroughStrokeWidth = 1,
  sampleTakenGuide: "none" | "line" | "all" = "none"
) {
  const g = container.group().addClass("bell");

  const inputRegions = calculateSubcloneRegions(tree, shapers, 0);

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
    if (Math.abs(upper2 - lower2) < 0.001) {
      return;
    }

    let element;

    // Round to one decimal place to make the SVG smaller
    const r = (x: number) => Math.round(x * 10) / 10;

    if (upper1 == upper2 && lower1 == lower2) {
      const inputRegion = inputRegions.get(node.subclone);
      if (inputRegion[1] - inputRegion[0] > 0.001) {
        // Seems to be a rectangle.
        element = g
          .rect(r(width), r((lower1 - upper1) * height))
          .move(0, r(upper1 * height));
      }
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

    if (element) {
      const color = subcloneColors.get(node.subclone);
      element
        .fill(color)
        .stroke({
          color: d3
            .color(color)
            .darker(bellPlotProperties.bellStrokeDarkening)
            .toString(),
          width: bellPlotProperties.bellStrokeWidth ?? 1,
        })
        .addClass("subclone")
        .data("subclone", node.subclone);

      if (shaper.emerging) {
        element.data(
          "descendant-subclones",
          treeToNodeArray(node).map((n) => n.subclone)
        );
      }

      element.add(
        SVG(
          `<title>${getSubcloneTooltip(
            node.subclone,
            shaper.subcloneMetrics
          )}</title>`
        )
      );
    }

    for (const child of node.children) {
      drawNode(child);
    }
  }

  drawNode(tree);

  for (const subclone of passThroughSubclones) {
    const color = subcloneColors.get(subclone);
    const y = shapers.get(subclone)(1, 0) * height;
    // TODO: What if there's an initial size that is not zero or one?
    // In that case, a curve should be drawn using the shaper.
    g.line(0, y, width, y)
      .stroke({
        color: color,
        width: passThroughStrokeWidth,
        dasharray: "6,4",
      })
      .addClass("pass-through")
      .attr({ "pointer-events": "none" })
      .data("subclone", subclone);
  }

  if (sampleTakenGuide != "none") {
    const sw = bellPlotProperties.bellStrokeWidth ?? 1;
    const x = Math.round(width * bellPlotProperties.bellPlateauPos);
    g.line(x, sw, x, height - sw)
      .stroke({
        color: "black",
        width: sw,
        dasharray: [1, 4].map((x) => x * sw).join(","),
        linecap: "round",
        opacity: sampleTakenGuide == "all" ? 0.6 : 0.3,
      })
      .addClass("sample-taken-line");

    if (sampleTakenGuide == "all") {
      const x2 = x + (width - x) / 2;

      drawArrowAndLabel(g, x2 - 14, 12, x2, 5, "Sample taken").translate(
        0,
        height
      );
    }
  }

  return g;
}

const tooltipFormat = d3.format(".3f");

function getSubcloneTooltip(subclone: Subclone, metrics: SubcloneMetrics) {
  let title = `Subclone: ${subclone}`;
  if (metrics) {
    title += `\nClonal prevalence: ${tooltipFormat(metrics.clonalPrevalence)}`;
    title += `\nCancer cell fraction: ${tooltipFormat(
      metrics.cancerCellFraction
    )}`;
  }
  return title;
}

export type Shaper = ((x: number, y: number) => number) & {
  /**
   * Metrics for a convenient access
   */
  subcloneMetrics?: SubcloneMetrics;

  /**
   * True if this shaper represents a subclone that is first seen in this sample
   */
  emerging?: boolean;
};

/**
 * Creates shaper functions for each subclone. The shapers are nested, which
 * ensures that descendants always stay within the boundaries of their parents.
 */
export function treeToShapers(
  phylogenyRoot: PhylogenyNode,
  metricsMap: SubcloneMetricsMap,
  preEmergedSubclones: Set<Subclone>,
  props: BellPlotProperties,
  collapsedSubclones: Set<Subclone>,
  emergingRootSize: number
): Map<Subclone, Shaper> {
  function getDepth(node: PhylogenyNode): number {
    return (
      (node.children ?? [])
        .filter(
          (n) =>
            metricsMap.get(n.subclone).cancerCellFraction > 0.001 &&
            !collapsedSubclones.has(n.subclone)
        ) // TODO: epsilon
        .map((n) => getDepth(n))
        .reduce((a, b) => Math.max(a, b), 0) + 1
    );
  }

  const shapers: Map<Subclone, Shaper> = new Map();

  function traverse(
    node: PhylogenyNode,
    parentShaper: Shaper = (x, y) => y,
    fractionalDepth = 0,
    transformX: (x: number) => number = (x) => x
  ) {
    const parentNode = node.parent;
    const subclone = node.subclone;
    const metrics = metricsMap.get(subclone);
    const preEmerged = preEmergedSubclones.has(subclone);

    const remainingDepth = getDepth(node);

    const fractionalStep = preEmerged
      ? 0
      : (1 - fractionalDepth) / (remainingDepth + 1);

    if (parentNode && !collapsedSubclones.has(parentNode.subclone)) {
      fractionalDepth += fractionalStep;
    } else {
      // Root node has a special handling: no step is added
    }

    const parentSubclone = node.parent?.subclone;
    if (
      !preEmerged &&
      parentSubclone &&
      preEmergedSubclones.has(parentSubclone) &&
      emergingRootSize <= 0
    ) {
      // Get rid of some unnecessary empty space before the first bell
      const a = fractionalStep / 2;
      // Create a plateau at the end so that the right edge looks like
      // a stacked bar chart.
      const b = 1 / props.bellPlateauPos;
      transformX = (x) => x * (b - a) + a;
    }

    const fancy = (x: number) => {
      const tipShape = clamp(0, 0.9999, props.bellTipShape);
      const transformedX = transformX(x);
      let value = fancystep(fractionalDepth, 1, transformedX, tipShape);
      if (node == phylogenyRoot) {
        value +=
          smoothstep(1, fractionalDepth, transformedX) * emergingRootSize;
      }
      return value;
    };

    const noStep = preEmerged || collapsedSubclones.has(node.parent?.subclone);
    const shaper: Shaper = (x, y) =>
      parentShaper(
        x,
        // The fractionalChildDepth defines when the bell starts to appear.
        lerp(fancy(x), 1, noStep ? 1 : 0) *
          metrics.fractionOfParent *
          (y - 0.5) +
          0.5
      );
    shaper.subcloneMetrics = metrics;
    shaper.emerging = !preEmerged;

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
          let a =
            currentDepth >= 1 ? 1 : smoothstep(currentDepth, 1, transformX(x));
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
        fractionalDepth,
        transformX
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
      const childShaper = shapers.get(child.subclone);
      if (childShaper.subcloneMetrics.cancerCellFraction > 0) {
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
