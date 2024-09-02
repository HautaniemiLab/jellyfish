import { NODE_TYPES, SampleTreeNode } from "./sampleTree.js";
import { treeToNodeArray } from "./tree.js";
import { fisherYatesShuffle, SeededRNG } from "./utils.js";

export interface NodePosition {
  node: SampleTreeNode;
  top: number;
  height: number;
}

export interface LayoutProperties {
  sampleHeight: number;
  sampleWidth: number;
  inferredSampleHeight: number;
  gapHeight: number;
  sampleSpacing: number;
  columnSpacing: number;
  canvasWidth: number;
  canvasHeight: number;
  randomSeed: string;
  randomizationRounds: number;
}

function sampleTreeToColumns(sampleTree: SampleTreeNode) {
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

function calculateCost(
  stackedColumns: NodePosition[][],
  layoutProps: LayoutProperties
) {
  const crossingFactor = 10;
  const pathLengthFactor = 1;

  let cost = 0;

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

      paths.push([
        parentPosition.top + parentPosition.height / 2,
        position.top + position.height / 2,
      ]);
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

  for (const paths of columnPaths) {
    cost += getNumberOfCrossings(paths) * crossingFactor;
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

  cost +=
    (getTotalPathLength(columnPaths.flat()) /
      (layoutProps.sampleHeight + layoutProps.sampleSpacing)) *
    pathLengthFactor;

  return cost;
}

function stackColumn(column: SampleTreeNode[], layoutProps: LayoutProperties) {
  let top = 0;

  const heights = {
    [NODE_TYPES.REAL_SAMPLE]: layoutProps.sampleHeight,
    [NODE_TYPES.INFERRED_SAMPLE]: layoutProps.inferredSampleHeight,
    [NODE_TYPES.GAP]: layoutProps.gapHeight,
  };

  const positions = [];

  let previousType;

  for (const node of column) {
    const type = node.type;
    if (
      previousType &&
      type != NODE_TYPES.GAP &&
      previousType != NODE_TYPES.GAP
    ) {
      top += layoutProps.sampleSpacing;
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
  const last = positions.at(-1);
  const offset = -(last.top + last.height) / 2;

  for (const position of positions) {
    position.top += offset;
  }

  return positions;
}

const randomizedColumns = (() => {
  const rng = SeededRNG(randomSeed);
  return columns.map((column) => fisherYatesShuffle(column, rng));
})();

const optimizationResult = (() => {
  let bestResult;
  let bestCost = Infinity;

  const rng = SeededRNG(randomSeed);

  for (let round = 0; round < randomizationRounds; round++) {
    const permutedColumns = columns.map((column) =>
      fisherYatesShuffle(column, rng)
    );
    const stackedColumns = permutedColumns.map(stackColumn);

    const cost = calculateCost(stackedColumns);
    if (cost < bestCost) {
      bestResult = stackedColumns;
      bestCost = cost;
    }
  }

  return {
    stackedColumns: bestResult,
    cost: bestCost,
  };
})();
