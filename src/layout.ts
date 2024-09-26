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

export interface LayoutProperties extends BellPlotProperties {
  sampleHeight: number;
  sampleWidth: number;
  inferredSampleHeight: number;
  gapHeight: number;
  sampleSpacing: number;
  columnSpacing: number;
  tentacleWidth: number;
  tentacleSpacing: number;
  sampleFontSize: number;
  showLegend: boolean;
  phylogenyColorScheme: boolean;
  phylogenyHueOffset: number;
  sampleTakenGuide: "none" | "line" | "text";
}

export interface CostWeights {
  crossing: number;
  pathLength: number;
  orderMismatch: number;
  bundleMismatch: number;
  divergence: number;
}

export const DEFAULT_COST_WEIGHTS: CostWeights = {
  crossing: 10,
  pathLength: 2,
  orderMismatch: 2,
  divergence: 3,
  bundleMismatch: 4,
};

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
  sampleDistanceMatrix: number[][] | null,
  costWeights: CostWeights
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
    costWeights.crossing > 0
      ? columnPaths.reduce((acc, paths) => acc + getNumberOfCrossings(paths), 0)
      : 0;

  const totalPathLength =
    costWeights.pathLength > 0
      ? columnPaths.reduce((acc, paths) => acc + getTotalPathLength(paths), 0) /
        (layoutProps.sampleHeight + layoutProps.sampleSpacing)
      : 0;

  const totalOrderMismatch =
    costWeights.orderMismatch > 0
      ? stackedColumns.reduce(
          (acc, column) => acc + getOrderMismatch(column, false),
          0
        )
      : 0;

  const totalDivergenceMismatch =
    costWeights.divergence > 0
      ? stackedColumns.reduce(
          (acc, column) => acc + getDivergenceMismatch(column),
          0
        )
      : 0;

  const totalBundleMismatch =
    costWeights.bundleMismatch > 0
      ? stackedColumns.reduce(
          (acc, column) => acc + getOrderMismatch(column, true),
          0
        )
      : 0;

  return (
    totalCrossings * costWeights.crossing +
    totalPathLength * costWeights.pathLength +
    totalOrderMismatch * costWeights.orderMismatch +
    totalDivergenceMismatch * costWeights.divergence +
    totalBundleMismatch * costWeights.bundleMismatch
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
  costWeights: CostWeights = DEFAULT_COST_WEIGHTS,
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
      sampleDistanceMatrix,
      costWeights
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
