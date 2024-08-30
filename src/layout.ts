import { treeToNodeArray } from "./sampleTree.js";

function sampleTreeToColumns(sampleTree) {
  const nodes = treeToNodeArray(sampleTree);

  const maxRank = nodes
    .map((node) => node.rank)
    .reduce((a, b) => Math.max(a, b), 0);

  const rankColumns = [];
  for (let i = 0; i <= maxRank; i++) {
    rankColumns[i] = [];
  }

  for (const node of nodes) {
    rankColumns[node.rank].push(node);
  }

  return rankColumns;
}

function calculateCost(stackedColumns) {
  const crossingFactor = 10;
  const pathLengthFactor = 1;

  let cost = 0;

  const columnPaths = [];
  // Extract paths
  for (let i = 1; i < stackedColumns.length; i++) {
    const paths = [];
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
  function getNumberOfCrossings(paths) {
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

  function getTotalPathLength(paths) {
    let sum = 0;

    for (const path of paths) {
      const vertical = path[1] - path[0];
      const len = Math.sqrt(columnSpacing ** 2 + vertical ** 2);
      sum += len;
    }
    return sum;
  }

  cost +=
    (getTotalPathLength(columnPaths.flat()) / (sampleHeight + sampleSpacing)) *
    pathLengthFactor;

  return cost;
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
