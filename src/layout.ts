import { BellPlotProperties } from "./bellplot.js";
import { SampleId } from "./data.js";
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
}

export interface CostWeights {
  crossing: number;
  pathLength: number;
  orderMismatch: number;
  divergence: number;
}

export const DEFAULT_COST_WEIGHTS: CostWeights = {
  crossing: 10,
  pathLength: 2,
  orderMismatch: 2,
  divergence: 3,
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
  preferredOrders: Map<SampleId, number>,
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

    // TODO: Optimize, don't create intermediate arrays
    const indices = stackedColumn
      .map((pos) => pos.node.sample?.indexNumber)
      .filter((i) => i != null);

    for (let i = 0; i < indices.length - 1; i++) {
      const a = indices[i];
      const b = indices[i + 1];

      const ab = sampleDistanceMatrix[a][b];

      mismatch += ab;
    }

    return mismatch;
  }

  function getOrderMismatch(stackedColumn: NodePosition[]) {
    let mismatch = 0;
    let previousPreference: number;

    for (let i = 0; i < stackedColumn.length; i++) {
      const node = stackedColumn[i].node;
      const preference = preferredOrders.get(node.sample?.sample);
      if (preference != null) {
        if (previousPreference != null) {
          mismatch += Math.max(0, previousPreference - preference);
        }
        previousPreference = preference;
      }
    }

    return mismatch;
  }

  const totalCrossings = columnPaths.reduce(
    (acc, paths) => acc + getNumberOfCrossings(paths),
    0
  );

  const totalPathLength =
    columnPaths.reduce((acc, paths) => acc + getTotalPathLength(paths), 0) /
    (layoutProps.sampleHeight + layoutProps.sampleSpacing);

  const totalOrderMismatch = stackedColumns.reduce(
    (acc, column) => acc + getOrderMismatch(column),
    0
  );

  const totalDivergenceMismatch = stackedColumns.reduce(
    (acc, column) => acc + getDivergenceMismatch(column),
    0
  );

  return (
    totalCrossings * costWeights.crossing +
    totalPathLength * costWeights.pathLength +
    totalOrderMismatch * costWeights.orderMismatch +
    totalDivergenceMismatch * costWeights.divergence
  );
}

function stackColumn(column: SampleTreeNode[], layoutProps: LayoutProperties) {
  let top = 0;

  const heights = {
    [NODE_TYPES.REAL_SAMPLE]: layoutProps.sampleHeight,
    [NODE_TYPES.INFERRED_SAMPLE]: layoutProps.inferredSampleHeight,
    [NODE_TYPES.GAP]: layoutProps.gapHeight,
  };

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

    const height = heights[type];

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
  preferredOrders: Map<SampleId, number> = new Map(),
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
    y: bb.y + (bb.height - legendHeight) / 2,
    width: legendWidth,
    height: legendHeight,
  };
}
