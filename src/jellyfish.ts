import { G, Svg, SVG } from "@svgdotjs/svg.js";
import {
  createBellPlotGroup,
  BellPlotProperties,
  Shaper,
  calculateSubcloneRegions,
  treeToShapers,
} from "./bellplot.js";
import {
  createSampleTreeFromData,
  NODE_TYPES,
  SampleTreeNode,
} from "./sampleTree.js";
import { lerp } from "./utils.js";
import { DataTables, SampleId, Subclone } from "./data.js";
import { treeToNodeArray } from "./tree.js";
import * as d3 from "d3";
import {
  LayoutProperties,
  NodePosition,
  optimizeColumns,
  sampleTreeToColumns,
} from "./layout.js";
import { drawLegend } from "./legend.js";
import {
  buildPhylogenyTree,
  PhylogenyNode,
  rotatePhylogeny,
} from "./phylogeny.js";
import { getBoundingBox, Rect } from "./geometry.js";
import {
  calculateSubcloneMetrics,
  getProportionsBySamples,
  SubcloneMetricsMap,
} from "./composition.js";

function findNodesBySubclone(
  sampleTree: SampleTreeNode,
  metricsBySample: Map<SampleId, SubcloneMetricsMap>,
  subclone: Subclone
) {
  const involvedNodes: SampleTreeNode[] = [];

  function find(node: SampleTreeNode) {
    let count = 0;

    const sample = node.sample?.sample;
    if (sample && metricsBySample.get(sample)?.get(subclone).clusterSize > 0) {
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

function isInheritingSubclone(
  sampleTreeNode: SampleTreeNode,
  subclone: Subclone,
  metricsBySample: Map<SampleId, SubcloneMetricsMap>
) {
  let node = sampleTreeNode.parent;
  while (node) {
    if (
      node.sample &&
      metricsBySample.get(node.sample.sample).get(subclone).clusterSize > 0
    ) {
      return true;
    }
    node = node.parent;
  }
  return false;
}

type ShapersAndRegions = {
  shapers: Map<Subclone, Shaper>;
  inputRegions: Map<Subclone, [number, number]>;
  outputRegions: Map<Subclone, [number, number]>;
};

type ShapersAndRegionsBySample = Map<SampleId, ShapersAndRegions>;

function findSubcloneLCAs(
  sampleTree: SampleTreeNode,
  phylogenyRoot: PhylogenyNode,
  metricsBySample: Map<SampleId, SubcloneMetricsMap>
) {
  return new Map(
    treeToNodeArray(phylogenyRoot)
      .map((node) => node.subclone)
      .map((subclone) => [
        subclone,
        findNodesBySubclone(sampleTree, metricsBySample, subclone).at(-1),
      ])
  );
}

function generateMetricsForInferredSample(
  sample: SampleId,
  phylogenyRoot: PhylogenyNode,
  subcloneLCAs: Map<Subclone, SampleTreeNode>
) {
  const phylogenyIndex = d3.index(
    treeToNodeArray(phylogenyRoot),
    (d) => d.subclone
  );

  const subclones = new Set(
    Array.from(subcloneLCAs.entries())
      .filter(([, node]) => node.sample.sample == sample)
      .map(([subclone]) => subclone)
  );

  // Add all ancestors
  for (const subclone of subclones) {
    let s = phylogenyIndex.get(subclone);
    while (s) {
      subclones.add(s.subclone);
      s = s.parent;
    }
  }

  const proportion = 1 / subclones.size;

  return calculateSubcloneMetrics(
    phylogenyRoot,
    new Map(
      Array.from(subclones.values()).map((subclone) => [subclone, proportion])
    )
  );
}

function createShapersAndRegions(
  sampleTree: SampleTreeNode,
  phylogenyRoot: PhylogenyNode,
  metricsBySample: Map<SampleId, SubcloneMetricsMap>,
  props: BellPlotProperties
): ShapersAndRegionsBySample {
  const allSubclones = treeToNodeArray(phylogenyRoot).map((d) => d.subclone);

  const handleSample = (node: SampleTreeNode) => {
    const preEmergedSubclones = new Set(
      allSubclones.filter((subclone) =>
        isInheritingSubclone(node, subclone, metricsBySample)
      )
    );
    const metricsMap = metricsBySample.get(node.sample.sample);
    const shapers = treeToShapers(
      phylogenyRoot,
      metricsMap,
      preEmergedSubclones,
      props
    );

    const calculateRegions = (edge: 0 | 1) =>
      calculateSubcloneRegions(phylogenyRoot, metricsMap, shapers, edge);

    const inputRegions = calculateRegions(0);
    const outputRegions = calculateRegions(1);

    return {
      shapers,
      inputRegions,
      outputRegions,
    } as ShapersAndRegions;
  };

  return new Map(
    treeToNodeArray(sampleTree)
      .filter((node) => node.sample)
      .map((node) => [node.sample.sample, handleSample(node)])
  );
}

function findSubcloneRanks(subcloneLCAs: Map<Subclone, SampleTreeNode>) {
  return new Map(
    Array.from(subcloneLCAs.entries()).map(([subclone, node]) => [
      subclone,
      node.rank,
    ])
  );
}

export function tablesToJellyfish(
  tables: DataTables,
  layoutProps: LayoutProperties
) {
  const { ranks, samples, phylogeny, compositions } = tables;

  const sampleTree = createSampleTreeFromData(samples, ranks);
  const proportionsBySamples = getProportionsBySamples(compositions);

  let phylogenyRoot = buildPhylogenyTree(phylogeny);

  const subcloneMetricsBySample = new Map(
    treeToNodeArray(sampleTree)
      .filter((node) => node.type == NODE_TYPES.REAL_SAMPLE)
      .map((node) => node.sample.sample)
      .map((sample) => [
        sample,
        calculateSubcloneMetrics(
          phylogenyRoot,
          proportionsBySamples.get(sample)
        ),
      ])
  );

  const subcloneLCAs = findSubcloneLCAs(
    sampleTree,
    phylogenyRoot,
    subcloneMetricsBySample
  );

  // Rotate phylogeny so that the subclones are in the correct order, i.e., the
  // later the rank, the earlier in the phylogeny (in the children list).
  // This is necessary for the stacked bell plots to be drawn correctly.
  phylogenyRoot = rotatePhylogeny(
    phylogenyRoot,
    findSubcloneRanks(subcloneLCAs)
  );

  // Root is an inferred sample
  subcloneMetricsBySample.set(
    sampleTree.sample.sample,
    generateMetricsForInferredSample(
      sampleTree.sample.sample,
      phylogenyRoot,
      subcloneLCAs
    )
  );

  const shapersAndRegionsBySample = createShapersAndRegions(
    sampleTree,
    phylogenyRoot,
    subcloneMetricsBySample,
    layoutProps
  );

  const subcloneColors = new Map(phylogeny.map((d) => [d.subclone, d.color]));

  const nodesInColumns = sampleTreeToColumns(sampleTree);
  const { stackedColumns } = optimizeColumns(nodesInColumns, layoutProps);

  const placement = getNodePlacement(stackedColumns, 40, layoutProps);

  return drawJellyfishSvg(
    placement,
    phylogenyRoot,
    shapersAndRegionsBySample,
    subcloneColors,
    layoutProps
  );
}

function getNodePlacement(
  stackedColumns: NodePosition[][],
  padding: number,
  layoutProps: LayoutProperties
) {
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

  return nodeCoords;
}

interface TentacleBundle {
  outputNode: SampleTreeNode;
  inputNode: SampleTreeNode;
  /** From inputNode to outputNode */
  gaps: SampleTreeNode[];
  subclones: Subclone[];
}

function collectTentacles(
  nodes: Iterable<SampleTreeNode>,
  shapersAndRegionsBySample: ShapersAndRegionsBySample
): TentacleBundle[] {
  const bundles: TentacleBundle[] = [];

  for (const inputNode of nodes) {
    // If node has a parent, a tentacle should be drawn.
    if (!inputNode.sample || !inputNode.parent) {
      continue;
    }

    const { inputRegions } = shapersAndRegionsBySample.get(
      inputNode.sample.sample
    );

    const subclones = Array.from(inputRegions.entries())
      .filter(([, inputRegion]) => inputRegion[1] - inputRegion[0] > 0)
      .sort((a, b) => a[1][0] - b[1][0])
      .map(([subclone]) => subclone);

    const gaps = [];

    let outputNode = inputNode.parent;
    while (outputNode && !outputNode.sample) {
      // TODO: Check if the gap is actually a gap
      gaps.push(outputNode);
      outputNode = outputNode.parent;
    }

    bundles.push({
      outputNode,
      inputNode,
      gaps,
      subclones,
    });
  }

  return bundles;
}

type OutputReservationsBySample = Map<
  SampleTreeNode,
  Map<Subclone, SampleTreeNode[]>
>;

/**
 * Reserves slots for the output tentacles of each sample and subclone.
 * Reservations are needed so that the output tentacles can be spread nicely
 * within the output regions of the subclones. We need to know the exact
 * number of tentacles for each subclone and the order in which they appear.
 */
function makeOutputReservations(
  tentacleBundles: TentacleBundle[],
  nodePlacement: Map<SampleTreeNode, Rect>
) {
  const tentacles = [];
  for (const bundle of tentacleBundles) {
    const nextNode =
      bundle.gaps.length > 0 ? bundle.gaps.at(-1) : bundle.inputNode;
    for (const subclone of bundle.subclones) {
      tentacles.push({
        outputNode: bundle.outputNode,
        nextNode,
        subclone,
      });
    }
  }

  return d3.rollup(
    tentacles,
    (D) =>
      D.map((d) => d.nextNode).sort(
        (a, b) => nodePlacement.get(a).y - nodePlacement.get(b).y
      ),
    (d) => d.outputNode,
    (d) => d.subclone
  ) as OutputReservationsBySample;
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

function drawTentacles(
  nodePlacement: Map<SampleTreeNode, Rect>,
  tentacleBundles: TentacleBundle[],
  shapersAndRegionsBySample: ShapersAndRegionsBySample,
  subcloneColors: Map<Subclone, string>,
  layoutProps: LayoutProperties
) {
  const { tentacleSpacing } = layoutProps;

  const reservations = makeOutputReservations(tentacleBundles, nodePlacement);

  const tentacleGroup = new G().addClass("tentacle-group");

  // Draw tentacles
  for (const bundle of tentacleBundles) {
    const midpoint = d3.mean;

    // TODO: The tentacles should be in the same order as they are in the
    // phylogeny, i.e., the most ancestral at the bottom.

    const { subclones } = bundle;

    const sample = bundle.inputNode.sample;
    const { inputRegions } = shapersAndRegionsBySample.get(sample.sample);

    const tentacleCount = subclones.length;

    const bundleGroup = tentacleGroup
      .group()
      .addClass("tentacle-bundle")
      .data("sample", sample.sample);

    for (let i = 0; i < tentacleCount; i++) {
      let inputNode = bundle.inputNode;

      const subclone = subclones[i];

      let outputNode = inputNode.parent;

      let inputPoint =
        midpoint(inputRegions.get(subclone)) *
        nodePlacement.get(inputNode).height;

      const path = d3.pathRound(1);

      // Draw the path through all (possible) gaps
      while (outputNode) {
        const outputCoords = nodePlacement.get(outputNode);
        const inputCoords = nodePlacement.get(inputNode);

        const outputPoint = outputNode.sample
          ? midpoint(
              shapersAndRegionsBySample
                .get(outputNode.sample.sample)
                .outputRegions.get(subclone)
            ) * outputCoords.height
          : outputCoords.height / 2 +
            getTentacleOffset(i, tentacleCount, tentacleSpacing);

        let outputOffsetY = 0;
        const subcloneReservations = reservations
          .get(outputNode)
          ?.get(subclone);
        if (subcloneReservations?.length) {
          const index = subcloneReservations.indexOf(inputNode);
          if (index < 0) {
            throw new Error("Tentacle reservation not found");
          }
          const region = shapersAndRegionsBySample
            .get(outputNode.sample.sample)
            .outputRegions.get(subclone);
          const regionHeight = (region[1] - region[0]) * outputCoords.height;
          outputOffsetY =
            regionHeight *
            ((index + 1) / (subcloneReservations.length + 1) - 0.5);
        }

        const ox = outputCoords.x + outputCoords.width;
        const oy = outputCoords.y + outputPoint + outputOffsetY;

        const ix = inputCoords.x;
        const iy = inputCoords.y + inputPoint;

        //const oMid = outputCoords.y + outputCoords.height / 2;
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
          ox - (outputNode.type == NODE_TYPES.GAP ? outputCoords.width / 2 : 0),
          oy
        );

        inputPoint = outputPoint;

        inputNode = outputNode;
        // Stop when the parent sample was found
        outputNode = outputNode.sample ? null : outputNode.parent;
      }

      bundleGroup
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
  return tentacleGroup;
}

function drawSamples(
  nodePlacement: Map<SampleTreeNode, Rect>,
  phylogenyRoot: PhylogenyNode,
  shapersAndRegionsBySample: ShapersAndRegionsBySample,
  subcloneColors: Map<Subclone, string>,
  layoutProps: LayoutProperties
) {
  const sampleGroup = new G().addClass("sample-group");

  for (const [node, coords] of nodePlacement.entries()) {
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

    const { shapers } = shapersAndRegionsBySample.get(sampleName);
    const bell = createBellPlotGroup(
      phylogenyRoot,
      shapers,
      subcloneColors,
      coords.width,
      coords.height
    );
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
  }
  return sampleGroup;
}

function drawJellyfishSvg(
  nodePlacement: Map<SampleTreeNode, Rect>,
  phylogenyRoot: PhylogenyNode,
  shapersAndRegionsBySample: ShapersAndRegionsBySample,
  subcloneColors: Map<Subclone, string>,
  layoutProps: LayoutProperties,
  padding = 40
): Svg {
  const legendWidth = layoutProps.showLegend ? 35 : 0; // TODO: Configurable

  const bb = getBoundingBox(nodePlacement.values());
  const canvasWidth = bb.width + 2 * padding + legendWidth;
  const canvasHeight = bb.height + 2 * padding;

  const svg = SVG().size(canvasWidth, canvasHeight);

  const rootGroup = svg.group().translate(0, canvasHeight / 2);

  rootGroup.add(
    drawSamples(
      nodePlacement,
      phylogenyRoot,
      shapersAndRegionsBySample,
      subcloneColors,
      layoutProps
    )
  );

  const tentacleBundles = collectTentacles(
    nodePlacement.keys(),
    shapersAndRegionsBySample
  );

  rootGroup.add(
    drawTentacles(
      nodePlacement,
      tentacleBundles,
      shapersAndRegionsBySample,
      subcloneColors,
      layoutProps
    )
  );

  if (layoutProps.showLegend) {
    const legend = drawLegend(subcloneColors);
    // TODO: A sophisticated way to position the legend
    legend.translate(canvasWidth - legendWidth, canvasHeight / 2);
    svg.add(legend);
  }

  return svg;
}
