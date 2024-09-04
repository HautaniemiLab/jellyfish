import { Svg, SVG } from "@svgdotjs/svg.js";
import {
  createBellPlotGroup,
  BellPlotNode,
  BellPlotProperties,
  createBellPlotTree,
  Shaper,
  calculateSubcloneRegions,
  treeToShapers,
} from "./bellplot.js";
import {
  createSampleTreeFromData,
  NODE_TYPES,
  SampleTreeNode,
} from "./sampleTree.js";
import { lerp, mapUnion } from "./utils.js";
import {
  DataTables,
  PhylogenyRow,
  SampleId,
  Subclone,
  getProportionsBySamples,
} from "./data.js";
import { stratify, TreeNode, treeToNodeArray } from "./tree.js";
import * as d3 from "d3";
import {
  LayoutProperties,
  NodePosition,
  optimizeColumns,
  sampleTreeToColumns,
} from "./layout.js";
import { createLegend } from "./legend.js";

type ProportionsBySamples = Map<string, Map<Subclone, number>>;

function findNodesBySubclone(
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

function isSampleInheritingSubclone(
  node: SampleTreeNode,
  subclone: Subclone,
  proportionsBySamples: ProportionsBySamples
) {
  node = node.parent;
  while (node) {
    if (
      node.sample &&
      proportionsBySamples.get(node.sample.sample).get(subclone) > 0
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

export interface PhylogenyNode extends TreeNode<PhylogenyNode> {
  subclone: Subclone;
}

function computeProportionsForInferredSamples(
  sampleTree: SampleTreeNode,
  proportionsBySamples: ProportionsBySamples,
  allSubclones: Subclone[],
  phylogenyTable: PhylogenyRow[]
) {
  const inferredProportions: ProportionsBySamples = new Map();

  const phylogenyRoot = stratify(
    phylogenyTable,
    (d) => d.subclone,
    (d) => d.parent,
    (d) =>
      ({
        subclone: d.subclone,
        parent: null,
        children: [],
      } as PhylogenyNode)
  );

  const nodesBySubclone = new Map(
    allSubclones.map((subclone) => [
      subclone,
      findNodesBySubclone(sampleTree, proportionsBySamples, subclone),
    ])
  );

  const subcloneLCAs = new Map(
    Array.from(nodesBySubclone.entries()).map(([subclone, nodes]) => [
      subclone,
      nodes.at(-1),
    ])
  );

  const inferredSamples = treeToNodeArray(sampleTree)
    .filter((node) => node.type == NODE_TYPES.INFERRED_SAMPLE)
    .map((node) => node.sample.sample);

  for (const sampleName of inferredSamples) {
    const subclones = new Set(
      Array.from(subcloneLCAs.entries())
        .filter(([, node]) => node?.sample?.sample == sampleName)
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

    inferredProportions.set(
      sampleName,
      new Map([...subclones.values()].map((subclone) => [subclone, proportion]))
    );
  }

  return inferredProportions;
}

function createBellPlotTreesAndShapers(
  sampleTree: SampleTreeNode,
  proportionsBySamples: ProportionsBySamples,
  phylogenyTable: PhylogenyRow[],
  allSubclones: Subclone[],
  props: BellPlotProperties
): BellPlotTreesAndShapers {
  return new Map(
    treeToNodeArray(sampleTree)
      .filter((node) => node.sample)
      .map((node) => {
        const tree = createBellPlotTree(
          phylogenyTable,
          proportionsBySamples.get(node.sample.sample),
          allSubclones.filter((subclone) =>
            isSampleInheritingSubclone(node, subclone, proportionsBySamples)
          )
        );
        const shapers = treeToShapers(tree, props);
        return [
          node.sample.sample,
          {
            tree,
            shapers,
            inputRegions: calculateSubcloneRegions(tree, shapers, 0),
            outputRegions: calculateSubcloneRegions(tree, shapers, 1),
          },
        ];
      })
  );
}

function findAllSubclones(
  proportionsBySamples: Map<SampleId, Map<Subclone, number>>
): Set<Subclone> {
  return new Set(
    Array.from(proportionsBySamples.values())
      .map((m) => Array.from(m.keys()))
      .flatMap((a) => a)
  );
}

export function tablesToJellyfish(
  tables: DataTables,
  layoutProps: LayoutProperties
) {
  const { ranks, samples, phylogeny, compositions } = tables;

  const sampleTree = createSampleTreeFromData(samples, ranks);

  const nodesInColumns = sampleTreeToColumns(sampleTree);
  const { stackedColumns } = optimizeColumns(nodesInColumns, layoutProps);

  const proportionsBySamples = getProportionsBySamples(compositions);

  const allSubclones = Array.from(new Set(phylogeny.map((d) => d.subclone)));

  const inferredProportions = computeProportionsForInferredSamples(
    sampleTree,
    proportionsBySamples,
    allSubclones,
    phylogeny
  );

  const treesAndShapers = createBellPlotTreesAndShapers(
    sampleTree,
    mapUnion(proportionsBySamples, inferredProportions),
    phylogeny,
    allSubclones,
    layoutProps
  );

  const subcloneColors = new Map(phylogeny.map((d) => [d.subclone, d.color]));

  return createJellyfishSvg(
    stackedColumns,
    treesAndShapers,
    subcloneColors,
    layoutProps
  );
}

const getTentacleOffset = (
  i: number,
  tentacleCount: number,
  tentacleSpacing: number,
  vec: number[] = [1, 0]
) =>
  (i - tentacleCount / 2 + 0.5) *
  tentacleSpacing *
  Math.abs(Math.sqrt(vec[0] ** 2 + vec[1] ** 2) / vec[0]);

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Returns a bounding box for a collection of rectangles.
 */
function getBoundingBox(rects: Iterable<Rect>) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const rect of rects) {
    minX = Math.min(minX, rect.x);
    minY = Math.min(minY, rect.y);
    maxX = Math.max(maxX, rect.x + rect.width);
    maxY = Math.max(maxY, rect.y + rect.height);
  }

  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function createJellyfishSvg(
  stackedColumns: NodePosition[][],
  bellPlotTreesAndShapers: BellPlotTreesAndShapers,
  subcloneColors: Map<Subclone, string>,
  layoutProps: LayoutProperties
): Svg {
  const padding = 40; // TODO: Configurable
  const legendWidth = layoutProps.showLegend ? 35 : 0; // TODO: Configurable

  const { tentacleSpacing } = layoutProps;

  const columnCount = stackedColumns.length;
  const columnPositions = [];
  for (let i = 0; i < columnCount; i++) {
    columnPositions.push({
      left: (layoutProps.sampleWidth + layoutProps.columnSpacing) * i + padding,
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

  const bb = getBoundingBox(nodeCoords.values());
  const canvasWidth = bb.width + 2 * padding + legendWidth;
  const canvasHeight = bb.height + 2 * padding;

  const svg = SVG().size(canvasWidth, canvasHeight);

  const rootGroup = svg.group().translate(0, canvasHeight / 2);
  const sampleGroup = rootGroup.group().addClass("sample-group");
  const tentacleGroup = rootGroup.group().addClass("tentacle-group");

  for (const [node, coords] of nodeCoords.entries()) {
    const sample = node.sample;

    if (!sample) {
      // Skip gaps, etc.
      continue;
    }

    const group = sampleGroup
      .group()
      .translate(coords.x, coords.y)
      .addClass("sample")
      .data("sample", sample.sample);

    const sampleName = sample.sample;
    const { tree, shapers, inputRegions } =
      bellPlotTreesAndShapers.get(sampleName);

    const bell = createBellPlotGroup(
      tree,
      shapers,
      subcloneColors,
      coords.width,
      coords.height
    ).addClass("bell");
    group.add(bell);

    const title = sample.displayName ?? sample.sample;
    group
      .text(title)
      .dx(coords.width / 2)
      .dy(-6)
      .font({
        family: "sans-serif",
        size: layoutProps.sampleFontSize,
        anchor: "middle",
      })
      .addClass("sample-display-name");

    // Draw tentacles

    const midpoint = (tuple: number[]) => (tuple[0] + tuple[1]) / 2;

    // If node has a parent, a tentacle should be drawn.
    // TODO: The tentacles should be in the same order as they are in the
    // phylogeny, i.e., the most ancestral at the bottom.
    if (node.parent) {
      // The following subclones need incoming tentacles
      const subclones = Array.from(inputRegions.entries())
        .filter(([, inputRegion]) => inputRegion[1] - inputRegion[0] > 0)
        .sort((a, b) => a[1][0] - b[1][0])
        .map(([subclone]) => subclone);

      console.log("sample", node.sample.sample, "incoming", subclones);

      const tentacleCount = subclones.length;

      const tentacleBundle = tentacleGroup
        .group()
        .addClass("tentacle-bundle")
        .data("sample", sample.sample);

      for (let i = 0; i < tentacleCount; i++) {
        const subclone = subclones[i];

        let inputNode = node;
        let outputNode = node.parent;

        let inputPoint = midpoint(inputRegions.get(subclone)) * coords.height;

        const path = d3.pathRound(1);

        // Draw the path through all (possible) gaps
        while (outputNode) {
          const outputCoords = nodeCoords.get(outputNode);
          const inputCoords = nodeCoords.get(inputNode);

          const outputPoint = outputNode.sample
            ? midpoint(
                bellPlotTreesAndShapers
                  .get(outputNode.sample.sample)
                  .outputRegions.get(subclone)
              ) * outputCoords.height
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
            width: layoutProps.tentacleWidth,
          })
          .attr({ "stroke-linecap": "square" })
          .fill("transparent")
          .addClass("tentacle")
          .data("subclone", subclone);
      }
    }
  }

  if (layoutProps.showLegend) {
    const legend = createLegend(subcloneColors);
    // TODO: A sophisticated way to position the legend
    legend.translate(canvasWidth - legendWidth, canvasHeight / 2);
    svg.add(legend);
  }

  return svg;
}
