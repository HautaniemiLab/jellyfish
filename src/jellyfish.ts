import { Svg, SVG } from "@svgdotjs/svg.js";
import {
  addTreeToSvgGroup,
  BellPlotNode,
  createBellPlotTree,
  Shaper,
  stackTree,
  treeToShapers,
} from "./bellplot.js";
import { NODE_TYPES, SampleTreeNode } from "./sampleTree.js";
import { lerp } from "./utils.js";
import { PhylogenyRow, SampleId, Subclone } from "./data.js";
import { treeToNodeArray } from "./tree.js";
import * as d3 from "d3";
import { LayoutProperties, NodePosition } from "./layout.js";

type ProportionsBySamples = Map<string, Map<Subclone, number>>;

export function findNodesBySubclone(
  sampleTree: SampleTreeNode,
  proportionsBySamples: ProportionsBySamples,
  subclone: Subclone
) {
  const involvedNodes: SampleTreeNode[] = [];

  function find(node: SampleTreeNode) {
    let count = 0;

    const sample = node.sample?.sample;
    if (sample && proportionsBySamples.get(sample)?.get(subclone) > 0) {
      involvedNodes.push(node);
      return 1;
    }

    for (const child of node.children) {
      count += find(child);
    }

    if (count > 1) {
      involvedNodes.push(node);
    }

    return count > 0 ? 1 : 0;
  }

  find(sampleTree);

  // Because of DFS, the last element is the Lowest Common Ancestor
  return involvedNodes;
}

/**
 * TODO: Rename. BellPlot is a bit misleading.
 */
function isBellPlotNodeInheritingSubclone(
  node: SampleTreeNode,
  subclone: Subclone,
  proportionsBySamples: ProportionsBySamples
) {
  node = node.parent;
  while (node) {
    if (
      node.sample &&
      proportionsBySamples.get(node.sample.sample)?.get(subclone) > 0
    ) {
      return true;
    }
    node = node.parent;
  }
  return false;
}

type BellPlotTreesAndShapers = Map<
  SampleId,
  {
    tree: BellPlotNode;
    shapers: Map<Subclone, Shaper>;
    inputRegions: Map<Subclone, [number, number]>;
    outputRegions: Map<Subclone, [number, number]>;
  }
>;

export function createBellPlotTreesAndShapers(
  sampleTree: SampleTreeNode,
  proportionsBySamples: ProportionsBySamples,
  phylogenyTable: PhylogenyRow[],
  allSubclones: Subclone[],
  subcloneLCAs: Map<Subclone, SampleTreeNode>
): BellPlotTreesAndShapers {
  const allProportions = new Map(proportionsBySamples);

  const nodes = treeToNodeArray(sampleTree);

  const inferredSamples = nodes
    .filter((node) => node.type == NODE_TYPES.INFERRED_SAMPLE)
    .map((node) => node.sample.sample);

  for (const sampleName of inferredSamples) {
    const subclones = new Set(
      [...subcloneLCAs.entries()]
        .filter(([, node]) => node.sample?.sample == sampleName)
        .map(([subclone]) => subclone)
    );

    // Add all ancestors
    for (const subclone of subclones) {
      let s = subclone;
      do {
        // TODO: use d3.index
        s = phylogenyTable.find((row) => row.subclone == s)?.parent;
        subclones.add(s);
      } while (s != null);
    }

    const proportion = 1 / subclones.size;

    allProportions.set(
      sampleName,
      new Map([...subclones.values()].map((subclone) => [subclone, proportion]))
    );
  }

  return new Map(
    nodes
      .filter((node) => node.sample)
      .map((node) => {
        const tree = createBellPlotTree(
          phylogenyTable,
          allProportions.get(node.sample.sample),
          allSubclones.filter((subclone) =>
            isBellPlotNodeInheritingSubclone(node, subclone, allProportions)
          )
        );
        const shapers = treeToShapers(tree);
        return [
          node.sample.sample,
          {
            tree,
            shapers,
            inputRegions: stackTree(tree, shapers, 0),
            outputRegions: stackTree(tree, shapers, 1),
          },
        ];
      })
  );
}

const getTentacleOffset = (
  i: number,
  tentacleCount: number,
  tentacleSpacing: number,
  vec: number[] = [1, 0]
) =>
  (i - tentacleCount / 2) *
  tentacleSpacing *
  Math.abs(Math.sqrt(vec[0] ** 2 + vec[1] ** 2) / vec[0]);

export function createBellPlotSvg(
  stackedColumns: NodePosition[][],
  bellPlotTreesAndShapers: BellPlotTreesAndShapers,
  subcloneColors: Map<Subclone, string>,
  layoutProps: LayoutProperties
): Svg {
  const leftPadding = 20;
  const tentacleSpacing = 4; // TODO: Configurable

  const columnCount = stackedColumns.length;
  const columnPositions = [];
  for (let i = 0; i < columnCount; i++) {
    columnPositions.push({
      left:
        (layoutProps.sampleWidth + layoutProps.columnSpacing) * i + leftPadding,
      width: layoutProps.sampleWidth,
    });
  }

  const nodeCoords = new Map();

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
      });
    }
  }

  const svg = SVG().size(layoutProps.canvasWidth, layoutProps.canvasHeight);

  const rootGroup = svg.group().translate(0, layoutProps.canvasHeight / 2);
  const sampleGroup = rootGroup.group().addClass("sample-group");
  const tentacleGroup = rootGroup.group().addClass("tentacle-group");

  for (const [node, coords] of nodeCoords.entries()) {
    const sample = node.sample;

    if (!sample) {
      // Skip gaps, etc.
      continue;
    }
    console.log("Sample:", sample);

    const group = sampleGroup
      .group()
      .translate(coords.x, coords.y)
      .addClass("sample")
      .data("sample", sample.sample);

    const bellGroup = group.group().addClass("bell");
    const sampleName = sample.sample;
    const { tree, shapers, inputRegions, outputRegions } =
      bellPlotTreesAndShapers.get(sampleName);

    addTreeToSvgGroup(tree, shapers, bellGroup, coords.width, coords.height);

    const title = sample.displayName ?? sample.sample;
    group
      .text(title)
      .dx(coords.width / 2)
      .dy(-6)
      .font({ family: "sans-serif", size: 12, anchor: "middle" })
      .addClass("sample-display-name");

    // Draw tentacles

    const midpoint = (tuple: number[]) => (tuple[0] + tuple[1]) / 2;

    // If node has a parent, a tentacle should be drawn.
    // TODO: The tentacles should be in the same order as they are in the
    // phylogeny, i.e., the most ancestral at the bottom.
    if (node.parent) {
      // The following subclones need incoming tentacles
      const subclones = [...inputRegions.entries()]
        .filter(([, inputRegion]) => inputRegion[1] - inputRegion[0] > 0)
        .sort((a, b) => a[1][0] - b[1][0])
        .map(([subclone]) => subclone);

      const tentacleCount = subclones.length;

      const tentacleBundle = tentacleGroup
        .group()
        .addClass("tentacle-bundle")
        .data("sample", sample.sample);

      for (let i = 0; i < tentacleCount; i++) {
        const subclone = subclones[i];

        console.log("subclone", subclone);

        let inputNode = node;
        let outputNode = node.parent;

        let inputPoint = midpoint(inputRegions.get(subclone)) * coords.height;

        let inputColumnPosition = columnPositions[node.rank];

        const path = d3.pathRound(1);

        // Draw the path through all (possible) gaps
        while (outputNode) {
          console.log("outputNode", outputNode);

          const outputCoords = nodeCoords.get(outputNode);
          const inputCoords = nodeCoords.get(inputNode);

          const outputPoint = outputNode.sample
            ? midpoint(outputRegions.get(subclone)) * outputCoords.height
            : outputCoords.height / 2 +
              getTentacleOffset(i, tentacleCount, tentacleSpacing);

          const ox = outputCoords.x + outputCoords.width;
          const oy = outputCoords.y + outputPoint;

          const ix = inputCoords.x;
          const iy = inputCoords.y + inputPoint;

          const oMid = outputCoords.y + outputCoords.height / 2;
          const iMid = inputCoords.y + inputCoords.height / 2;

          // The distance as a fraction of column spacing
          const inputOutputCPDist = 0.3;
          // Position of bezier's control points (input and output)
          const ixc = lerp(ox, ix, 1 - inputOutputCPDist);
          const oxc = lerp(ox, ix, inputOutputCPDist);

          // Squeeze the bundle at the midpoint between the samples (in both x and y)
          const sx = (ox + ix) / 2;
          const sy =
            (outputCoords.y +
              inputCoords.y +
              (outputCoords.height + inputCoords.height) / 2) /
            2;

          const scDist = 0.6;

          const midXCpOffset = lerp(ix, sx, 1 - scDist * 0.5) - sx;
          const midYCpOffset = lerp(iMid, sy, 1 - scDist) - sy;

          const sControlOffset = getTentacleOffset(
            i,
            tentacleCount,
            tentacleSpacing,
            [midXCpOffset, midYCpOffset]
          );

          if (inputNode.sample) {
            path.moveTo(ix, iy);
          }
          // (cpx1, cpy1, cpx2, cpy2, x, y)
          // TODO: Squeezing is unnecessary if there's only a single tentacle
          path.bezierCurveTo(
            ixc,
            iy,
            sx + midXCpOffset,
            sy + midYCpOffset + sControlOffset,
            sx,
            sy + sControlOffset
          );
          path.bezierCurveTo(
            sx - midXCpOffset,
            sy - midYCpOffset + sControlOffset,
            oxc,
            oy,
            ox -
              (outputNode.type == NODE_TYPES.GAP ? outputCoords.width / 2 : 0),
            oy
          );

          inputPoint = outputPoint;

          inputNode = outputNode;
          // Stop when the parent sample was found
          outputNode = outputNode.sample ? null : outputNode.parent;
        }

        tentacleBundle
          .path(path.toString())
          .stroke({
            color: subcloneColors.get(subclone) ?? "black",
            width: 2,
          })
          .attr({ "stroke-linecap": "square" })
          .fill("transparent")
          .addClass("tentacle")
          .data("subclone", subclone);
      }
    }
  }

  return svg;
}
