import { SVG } from "@svgdotjs/svg.js";
import * as d3 from "d3";
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

export function optimizeColumns(
  columns: SampleTreeNode[][],
  layoutProps: LayoutProperties,
  random: () => number = SeededRNG(0),
  randomizationRounds: number = 1000
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

    const cost = calculateCost(stackedColumns, layoutProps);
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

export function columnsToSvg(
  stackedColumns: { node: SampleTreeNode; top: number; height: number }[][],
  layoutProps: LayoutProperties
) {
  const leftPadding = 20;

  const columnCount = stackedColumns.length;
  const columnPositions = [];
  for (let i = 0; i < columnCount; i++) {
    columnPositions.push({
      left:
        (layoutProps.sampleWidth + layoutProps.columnSpacing) * i + leftPadding,
      width: layoutProps.sampleWidth,
    });
  }

  const svg = SVG().size(layoutProps.canvasWidth, layoutProps.canvasHeight);

  const rootGroup = svg
    .group()
    .transform({ translateY: layoutProps.canvasHeight / 2 });
  const tentacleGroup = rootGroup.group();
  const sampleGroup = rootGroup.group();

  for (let i = 0; i < columnCount; i++) {
    const positions = stackedColumns[i];
    const columnPosition = columnPositions[i];

    for (let j = 0; j < positions.length; j++) {
      const position = positions[j];
      const group = sampleGroup.group().transform({
        translateX: columnPosition.left,
        translateY: position.top,
      });

      const node = position.node;

      const sample = position.node.sample;

      if (node.type != NODE_TYPES.GAP) {
        group.rect(layoutProps.sampleWidth, position.height).attr({
          stroke: "gray",
          fill: "#fafafa",
        });
      }

      if (sample) {
        const title = sample.displayName ?? sample.sample;
        group
          .text(title)
          .dx(layoutProps.sampleWidth / 2)
          .dy(-5)
          .font({ family: "sans-serif", size: 12, anchor: "middle" });
      }

      // If node has a parent, a tentacle should be drawn
      if (node.parent) {
        const parentPosition = stackedColumns[i - 1].find(
          (pos) => pos.node == node.parent
        );

        const parentColumnPosition = columnPositions[i - 1];

        let px = parentColumnPosition.left + parentColumnPosition.width;
        let py = parentPosition.top + parentPosition.height / 2;

        let x = columnPosition.left;
        let y = position.top + position.height / 2;

        // Position of bezier's control points
        let pMidX = (px + x) / 2;
        let midX = pMidX;

        // Make gap crossings seamless, adjust the endpoints and control points
        // TODO: Draw a continuous path over the gap, connecting the samples
        if (node.type == NODE_TYPES.GAP) {
          x += columnPosition.width / 2;
          midX -= columnPosition.width * 0.3;
        }
        if (node.parent.type == NODE_TYPES.GAP) {
          px -= parentColumnPosition.width / 2;
          pMidX += parentColumnPosition.width * 0.3;
        }

        const p = d3.path();
        p.moveTo(px, py);
        p.bezierCurveTo(pMidX, py, midX, y, x, y);

        tentacleGroup.path(p.toString()).attr({ stroke: "gray", fill: "none" });
      }
    }
  }

  return svg;
}
