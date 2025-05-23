import { BellPlotProperties } from "./bellplot.js";
import { getBoundingBox, isIntersecting, Rect } from "./geometry.js";
import { NODE_TYPES, SampleTreeNode } from "./sampleTree.js";
import { treeToNodeArray } from "./tree.js";
import { fisherYatesShuffle, SeededRNG } from "./utils.js";

export interface NodePosition {
  node: SampleTreeNode;
  top: number;
  height: number;
}

export interface LayoutProperties extends BellPlotProperties, CostWeights {
  /**
   * Height of real sample nodes
   *
   * @minimum 10
   * @default 110
   */
  sampleHeight: number;

  /**
   * Width of sample nodes
   *
   * @minimum 10
   * @default 90
   */
  sampleWidth: number;

  /**
   * Height of inferred sample nodes
   *
   * @minimum 10
   * @default 120
   */
  inferredSampleHeight: number;

  /**
   * Height of gaps between samples. Gaps are routes for tentacle bundles.
   *
   * @minimum 0
   * @default 60
   */
  gapHeight: number;

  /**
   * Vertical space between samples
   *
   * @minimum 0
   * @default 60
   */
  sampleSpacing: number;

  /**
   * Horizontal space between columns
   *
   * @minimum 10
   * @default 90
   */
  columnSpacing: number;

  /**
   * Width of tentacles in pixels
   *
   * @minimum 0
   * @default 2
   */
  tentacleWidth: number;

  /**
   * Space between tentacles in a bundle, in pixels
   *
   * @minimum 0
   * @default 5
   */
  tentacleSpacing: number;

  /**
   * Relative distance of tentacle control points from the edge of the sample node
   *
   * @minimum 0
   * @maximum 0.45
   * @default 0.3
   */
  inOutCPDistance: number;

  /**
   * Relative distance of tentacle bundle's control points. The higher the value,
   * the longer the individual tentacles stay together before diverging.
   *
   * @minimum 0
   * @maximum 1.2
   * @default 0.6
   */
  bundleCPDistance: number;

  /**
   * Font size for sample labels
   *
   * @minimum 0
   * @default 12
   */
  sampleFontSize: number;

  /**
   * Whether to show the legend

   * @default true
   */
  showLegend: boolean;

  /**
   * Whether to use a color scheme based on phylogeny
   *
   * @default true
   */
  phylogenyColorScheme: boolean;

  /**
   * Offset for the hue of the phylogeny color scheme. If the automatically generated
   * hues are not to your liking, you can adjust the hue offset to get a different
   * color scheme.
   *
   * @minimum 0
   * @maximum 360
   * @default 0
   */
  phylogenyHueOffset: number;

  /**
   * Type of the "sample taken" guide.
   *
   * `"none"` for no guides,
   * `"line"` for a faint dashed line in all samples,
   * `"text"` same as line, but with a text label in one of the samples.
   * `"text-all"` same as text, but with a text label in all samples.
   *
   * @default "text"
   */
  sampleTakenGuide: "none" | "line" | "text" | "text-all";

  /**
   * Whether to show rank titles above the samples (if provided).
   *
   * @default true
   */
  showRankTitles: boolean;

  /**
   * Whether the root of the phylogenetic tree contains normal cells. If true,
   * no tentacles will be drawn for the root clone and its color will be white
   * if phylogenyColorScheme is used.
   *
   * @default false
   */
  normalsAtPhylogenyRoot: boolean;
}

export interface CostWeights {
  /**
   * Weight for tentacle bundles between two pairs of samples crossing each other.
   *
   * @minimum 0
   * @default 10
   */
  crossingWeight: number;

  /**
   * Weight for the total length of the paths (tentacle bundles) connecting samples.
   *
   * @minimum 0
   * @default 2
   */
  pathLengthWeight: number;

  /**
   * Weight for the mismatch in the order of samples. The order is based on the
   * "phylogenetic center of mass" computed from the subclonal compositions.
   *
   * @minimum 0
   * @default 2
   */
  orderMismatchWeight: number;

  /**
   * Weight for the mismatch in the placement of bundles. The "optimal" placement is
   * based on the subclonal compositions, but such placement may produce excessively
   * long tentacle bundles.
   *
   * @minimum 0
   * @default 3
   */
  bundleMismatchWeight: number;

  /**
   * Weight for the sum of divergences between adjacent samples.
   *
   * @minimum 0
   * @default 4
   */
  divergenceWeight: number;
}

export function sampleTreeToColumns(sampleTree: SampleTreeNode) {
  const nodes = treeToNodeArray(sampleTree);

  const maxRank = nodes
    .map((node) => node.rank)
    .reduce((a, b) => Math.max(a, b), 0);

  const rankColumns: SampleTreeNode[][] = [];
  for (let i = 0; i <= maxRank; i++) {
    rankColumns[i] = [];
  }

  for (const node of nodes) {
    rankColumns[node.rank].push(node);
  }

  return rankColumns;
}

export function getNodePlacement(
  stackedColumns: NodePosition[][],
  layoutProps: LayoutProperties
) {
  const columnCount = stackedColumns.length;
  const columnPositions = [];
  for (let i = 0; i < columnCount; i++) {
    columnPositions.push({
      left: (layoutProps.sampleWidth + layoutProps.columnSpacing) * i,
      width: layoutProps.sampleWidth,
    });
  }

  const nodeCoords = new Map<SampleTreeNode, Rect>();

  for (let i = 0; i < columnCount; i++) {
    const positions = stackedColumns[i];
    const columnPosition = columnPositions[i];

    for (let j = 0; j < positions.length; j++) {
      const position = positions[j];
      nodeCoords.set(position.node, {
        x: columnPosition.left,
        y: position.top,
        width: columnPosition.width,
        height: position.height,
      } as Rect);
    }
  }

  return nodeCoords;
}

function calculateCost(
  stackedColumns: NodePosition[][],
  layoutProps: LayoutProperties,
  preferredOrders: number[],
  sampleDistanceMatrix: number[][] | null
) {
  const columnPaths: number[][][] = [];

  // Extract paths
  for (let i = 1; i < stackedColumns.length; i++) {
    const paths: number[][] = [];
    columnPaths.push(paths);

    const column = stackedColumns[i];
    const parentColumn = stackedColumns[i - 1];

    for (const position of column) {
      const node = position.node;

      const parentPosition = parentColumn.find(
        (pos) => pos.node == node.parent
      );

      if (parentPosition) {
        paths.push([
          parentPosition.top + parentPosition.height / 2,
          position.top + position.height / 2,
        ]);
      } else {
        // Should not happen. TODO: Fix this
      }
    }
  }

  // A naive algorithm to find crossings
  function getNumberOfCrossings(paths: number[][]) {
    let crossings = 0;
    for (const p1 of paths) {
      for (const p2 of paths) {
        if (p1 == p2) {
          continue;
        }

        if (p1[0] < p2[0] && p1[1] > p2[1]) {
          crossings += 1;
        }
      }
    }
    return crossings;
  }

  function getTotalPathLength(paths: number[][]) {
    let sum = 0;

    for (const path of paths) {
      const vertical = path[1] - path[0];
      const len = Math.sqrt(layoutProps.columnSpacing ** 2 + vertical ** 2);
      sum += len;
    }
    return sum;
  }

  function getDivergenceMismatch(stackedColumn: NodePosition[]) {
    let mismatch = 0;
    let previousIndex = -1;

    for (const pos of stackedColumn) {
      const currentIndex =
        pos.node.type == NODE_TYPES.REAL_SAMPLE
          ? pos.node.sample?.indexNumber
          : -1;

      if (currentIndex >= 0) {
        if (previousIndex >= 0) {
          mismatch += sampleDistanceMatrix[previousIndex][currentIndex];
        }
        previousIndex = currentIndex;
      }
    }

    return mismatch;
  }

  function findSample(sampleOrGap: SampleTreeNode) {
    while (sampleOrGap.type == NODE_TYPES.GAP) {
      sampleOrGap = sampleOrGap.children[0];
    }
    return sampleOrGap;
  }

  /**
   * @param gaps true if the mismatch should be calculated for gaps between samples
   */
  function getOrderMismatch(stackedColumn: NodePosition[], gaps: boolean) {
    let mismatch = 0;
    let previousPreference = -1;

    for (const { node } of stackedColumn) {
      const tracedNode = gaps ? findSample(node) : node;
      const preference = tracedNode.sample
        ? preferredOrders[tracedNode.sample.indexNumber]
        : -1;

      if (preference >= 0) {
        if (previousPreference >= 0 && gaps == (node.type == NODE_TYPES.GAP)) {
          mismatch += Math.max(0, previousPreference - preference);
        }
        if (gaps || (!gaps && node.type != NODE_TYPES.GAP)) {
          previousPreference = preference;
        }
      }
    }

    return mismatch;
  }

  const totalCrossings =
    layoutProps.crossingWeight > 0
      ? columnPaths.reduce((acc, paths) => acc + getNumberOfCrossings(paths), 0)
      : 0;

  const totalPathLength =
    layoutProps.pathLengthWeight > 0
      ? columnPaths.reduce((acc, paths) => acc + getTotalPathLength(paths), 0) /
        (layoutProps.sampleHeight + layoutProps.sampleSpacing)
      : 0;

  const totalOrderMismatch =
    layoutProps.orderMismatchWeight > 0
      ? stackedColumns.reduce(
          (acc, column) => acc + getOrderMismatch(column, false),
          0
        )
      : 0;

  const totalDivergenceMismatch =
    layoutProps.divergenceWeight > 0
      ? stackedColumns.reduce(
          (acc, column) => acc + getDivergenceMismatch(column),
          0
        )
      : 0;

  const totalBundleMismatch =
    layoutProps.bundleMismatchWeight > 0
      ? stackedColumns.reduce(
          (acc, column) => acc + getOrderMismatch(column, true),
          0
        )
      : 0;

  return (
    totalCrossings * layoutProps.crossingWeight +
    totalPathLength * layoutProps.pathLengthWeight +
    totalOrderMismatch * layoutProps.orderMismatchWeight +
    totalDivergenceMismatch * layoutProps.divergenceWeight +
    totalBundleMismatch * layoutProps.bundleMismatchWeight
  );
}

const heightProps: Record<string, keyof LayoutProperties> = {
  [NODE_TYPES.REAL_SAMPLE]: "sampleHeight",
  [NODE_TYPES.INFERRED_SAMPLE]: "inferredSampleHeight",
  [NODE_TYPES.GAP]: "gapHeight",
} as const;

function stackColumn(column: SampleTreeNode[], layoutProps: LayoutProperties) {
  let top = 0;

  const positions: NodePosition[] = [];

  let previousType;

  for (const node of column) {
    const type = node.type;
    if (previousType) {
      if (type != NODE_TYPES.GAP && previousType != NODE_TYPES.GAP) {
        top += layoutProps.sampleSpacing;
      } else if (type == NODE_TYPES.GAP && previousType == NODE_TYPES.GAP) {
        // Allow some overlap between gaps so that adjacent tentacle bundles
        // don't look too separated.
        top -= layoutProps.gapHeight / 2.5;
      }
    }

    const height = layoutProps[heightProps[type]] as number;

    positions.push({
      node,
      top,
      height,
    });

    top += height;

    previousType = type;
  }

  // Center around zero
  const offset = top / 2;
  for (const position of positions) {
    position.top -= offset;
  }

  return positions;
}

export function optimizeColumns(
  columns: SampleTreeNode[][],
  layoutProps: LayoutProperties,
  preferredOrders: number[],
  sampleDistanceMatrix: number[][] | null,
  random: () => number = SeededRNG(0),
  randomizationRounds: number = 10000
) {
  let bestResult;
  let bestCost = Infinity;

  for (let round = 0; round < randomizationRounds; round++) {
    const permutedColumns = columns.map((column) =>
      fisherYatesShuffle(column, random)
    );
    const stackedColumns = permutedColumns.map((column) =>
      stackColumn(column, layoutProps)
    );

    const cost = calculateCost(
      stackedColumns,
      layoutProps,
      preferredOrders,
      sampleDistanceMatrix
    );
    if (cost < bestCost) {
      bestResult = stackedColumns;
      bestCost = cost;
    }
  }

  return {
    stackedColumns: bestResult,
    cost: bestCost,
  };
}

export function findLegendPlacement(
  nodePlacement: Map<SampleTreeNode, Rect>,
  legendWidth: number,
  legendHeight: number
): Rect {
  const rects = Array.from(nodePlacement.values());

  // TODO: Configurable
  const vPadding = 30;
  const hPadding = 10;

  const paddedWidth = legendWidth + hPadding * 2;
  const paddedHeight = legendHeight + vPadding * 2;

  const bb = getBoundingBox(rects);

  const bottomRight = {
    x: bb.x + bb.width - paddedWidth,
    y: bb.y + bb.height - paddedHeight,
  };
  const topRight = {
    x: bb.x + bb.width - paddedWidth,
    y: bb.y,
  };
  const bottomLeft = {
    x: bb.x,
    y: bb.y + bb.height - paddedHeight,
  };
  const topLeft = {
    x: bb.x,
    y: bb.y,
  };

  for (const coords of [bottomRight, topRight, bottomLeft, topLeft]) {
    const paddedRect = { ...coords, width: paddedWidth, height: paddedHeight };
    if (!isIntersecting(paddedRect, rects)) {
      return {
        x: coords.x + hPadding,
        y: coords.y + vPadding,
        width: legendWidth,
        height: legendHeight,
      };
    }
  }

  // Default placement: next to the the plot
  return {
    x: bb.x + bb.width + 40,
    y: bb.y + Math.round((bb.height - legendHeight) / 2),
    width: legendWidth,
    height: legendHeight,
  };
}
