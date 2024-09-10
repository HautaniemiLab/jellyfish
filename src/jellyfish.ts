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
import { treeIterator, treeToNodeArray } from "./tree.js";
import * as d3 from "d3";
import {
  getNodePlacement,
  LayoutProperties,
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
  calculateCentresOfMass,
  calculateSubcloneMetrics,
  getProportionsBySamples,
  SubcloneMetricsMap,
} from "./composition.js";
import { createDistanceMatrix, jsDivergence } from "./statistics.js";

/**
 * This is the main function that glues everything together.
 */
export function tablesToJellyfish(
  tables: DataTables,
  layoutProps: LayoutProperties
) {
  const { ranks, samples, phylogeny, compositions } = tables;

  /**
   * A tree structure that represents the samples and their relationships.
   * Samples that are not in adjacent ranks are connected with gaps.
   */
  const sampleTree = createSampleTreeFromData(samples, ranks);

  /** All sample tree nodes in depth-first order. Just for easy iteration. */
  const nodeArray = treeToNodeArray(sampleTree);

  /** The subclonal compositions of the samples */
  const proportionsBySamples = getProportionsBySamples(compositions);

  /**
   * The phylogenetic tree of the tumor with subclones as nodes. N.B.,
   * we have two distinct trees: the sample tree and the phylogeny.
   */
  const phylogenyRoot = buildPhylogenyTree(phylogeny);

  /**
   * The metrics for each subclone, separately for each sample. This includes
   * the size (or fraction or VAF) of the subclone in the sample, the size of
   * the cluster (the subclone and its descendants), and the subclone size
   * scaled to its parent cluster's size.
   */
  const subcloneMetricsBySample = new Map(
    nodeArray
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

  /**
   * The Lowest Common Ancestor of each cluster (not subclone) in the sample tree.
   * This allows for placing subclones that appear in multiple samples to the
   * inferred samples.
   */
  const subcloneLCAs = findSubcloneLCAs(
    sampleTree,
    phylogenyRoot,
    subcloneMetricsBySample
  );

  /**
   * Rotated phylogeny so that the subclones are in the correct order, i.e., the
   * later the rank, the earlier the subclone is in the phylogeny (in the children list).
   * This is necessary for the stacked bell plots to be drawn correctly.
   */
  const rotatedPhylogenyRoot = rotatePhylogeny(
    phylogenyRoot,
    findSubcloneRanks(subcloneLCAs)
  );

  // Root is an inferred sample. Subclones that are present in multiple samples in
  // its immediate children are placed in the inferred sample.
  subcloneMetricsBySample.set(
    sampleTree.sample.sample,
    generateMetricsForInferredSample(
      sampleTree.sample.sample,
      rotatedPhylogenyRoot,
      subcloneLCAs
    )
  );

  /**
   * Nodes in columns, i.e., the samples and gaps in each rank. The initial order
   * is very likely to be suboptimal.
   */
  const nodesInColumns = sampleTreeToColumns(sampleTree);

  /**
   * The centre of mass based on samples' subclonal compositions. This allows
   * for finding an optimal order for the samples within the ranks.
   */
  const centresOfMass = calculateCentresOfMass(
    rotatedPhylogenyRoot,
    subcloneMetricsBySample
  );

  /**
   * The distance matrix between the samples based on their subclonal compositions.
   * This is used to find an optimal order for the samples within the ranks, i.e.,
   * samples that are more similar are placed closer to each other.
   */
  const sampleDistanceMatrix = computeSubclonalDivergence(
    nodeArray,
    subcloneMetricsBySample
  );

  /**
   * Find an optimal (or good) permutation of the samples within each rank.
   * As a result, we get an order and vertical coordinates for the nodes.
   */
  const { stackedColumns } = optimizeColumns(
    nodesInColumns,
    layoutProps,
    centresOfMass,
    sampleDistanceMatrix,
    {
      crossing: 10,
      pathLength: 1,
      orderMismatch: 1,
      divergence: 1.5,
    }
  );

  /**
   * X/Y coordinates and width/height for each node. The placement can be
   * adjusted here, if needed.
   */
  const placement = getNodePlacement(stackedColumns, 40, layoutProps);

  /**
   * Shapers are functions that define the shape of the subclones in bell plots.
   * The regions define the attachment points for the tentacles.
   */
  const shapersAndRegionsBySample = createShapersAndRegions(
    sampleTree,
    rotatedPhylogenyRoot,
    subcloneMetricsBySample,
    layoutProps
  );

  const passThroughSubclones = findPassThroughSubclonesBySamples(
    sampleTree,
    shapersAndRegionsBySample,
    subcloneLCAs
  );

  const subcloneColors = new Map(phylogeny.map((d) => [d.subclone, d.color]));

  return drawJellyfishSvg(
    placement,
    rotatedPhylogenyRoot,
    shapersAndRegionsBySample,
    passThroughSubclones,
    subcloneColors,
    layoutProps
  );
}

// ----------------------------------------------------------------------------

/**
 * Finds samples and subclones that are pass-through, i.e., they are not
 * present in the sample but are present in an ancestor and a descendant.
 */
function findPassThroughSubclonesBySamples(
  sampleTree: SampleTreeNode,
  shapersAndRegionsBySample: ShapersAndRegionsBySample,
  subcloneLCAs: Map<Subclone, SampleTreeNode>
) {
  // TODO: This uses input and output regions, which works, but is not very
  // elegant. It could alternatively use subclone and cluster sizes directly.

  const inputsBySample = new Map<SampleId, Set<Subclone>>();
  const outputsBySample = new Map<SampleId, Set<Subclone>>();
  for (const [
    sample,
    { inputRegions, outputRegions },
  ] of shapersAndRegionsBySample) {
    inputsBySample.set(sample, new Set(getSubclonesFromRegions(inputRegions)));
    outputsBySample.set(
      sample,
      new Set(getSubclonesFromRegions(outputRegions))
    );
  }

  const result = new Map<SampleId, Set<Subclone>>();
  for (const node of treeIterator(sampleTree)) {
    const sample = node.sample?.sample;
    if (sample) {
      result.set(sample, new Set());
    }
  }

  for (const node of treeIterator(sampleTree)) {
    if (!node.sample || !node.parent) {
      continue;
    }

    for (const subclone of inputsBySample.get(node.sample.sample)) {
      const lca = subcloneLCAs.get(subclone);
      let n = node.parent;
      while (n && n != lca) {
        if (n.type == NODE_TYPES.GAP) {
          n = n.parent;
          continue;
        }

        if (
          outputsBySample.get(n.sample.sample).has(subclone) ||
          inputsBySample.get(n.sample.sample).has(subclone)
        ) {
          break;
        }

        result.get(n.sample.sample).add(subclone);

        n = n.parent;
      }
    }
  }

  return result;
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
  function findLCA(subclone: Subclone) {
    let lca: SampleTreeNode;

    function find(node: SampleTreeNode) {
      let count = 0;

      const sample = node.sample?.sample;
      if (
        sample &&
        metricsBySample.get(sample)?.get(subclone).clusterSize > 0
      ) {
        lca = node;
        return 1;
      }

      for (const child of node.children) {
        count += find(child);
      }

      if (count > 1) {
        lca = node;
      }

      return count > 0 ? 1 : 0;
    }

    find(sampleTree);

    // Because of DFS, the last element is the Lowest Common Ancestor
    return lca;
  }

  return new Map(
    treeToNodeArray(phylogenyRoot)
      .map((node) => node.subclone)
      .map((subclone) => [subclone, findLCA(subclone)])
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

function computeSubclonalDivergence(
  nodes: SampleTreeNode[],
  subcloneMetricsBySample: Map<SampleId, SubcloneMetricsMap>
) {
  const distributionsBySampleIndex = nodes
    .filter((node) => node.sample?.indexNumber != null)
    .sort((a, b) => a.sample.indexNumber - b.sample.indexNumber)
    .map((node) =>
      Array.from(subcloneMetricsBySample.get(node.sample.sample).values()).map(
        (metrics) => metrics.subcloneSize
      )
    );
  return createDistanceMatrix(distributionsBySampleIndex, jsDivergence);
}

interface TentacleBundle {
  outputNode: SampleTreeNode;
  inputNode: SampleTreeNode;
  /** From inputNode to outputNode */
  gaps: SampleTreeNode[];
  subclones: Subclone[];
}

function getSubclonesFromRegions(
  inputRegions: Map<Subclone, [number, number]>,
  extraSubclones: Set<Subclone> = new Set()
) {
  return Array.from(inputRegions.entries())
    .filter(
      ([subclone, inputRegion]) =>
        inputRegion[1] - inputRegion[0] > 0 || extraSubclones.has(subclone)
    )
    .sort((a, b) => a[1][0] - b[1][0] + a[1][1] - b[1][1])
    .map(([subclone]) => subclone);
}

function collectTentacles(
  nodes: Iterable<SampleTreeNode>,
  shapersAndRegionsBySample: ShapersAndRegionsBySample,
  passThroughSubclones: Map<SampleId, Set<Subclone>>
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

    const subclones = getSubclonesFromRegions(
      inputRegions,
      passThroughSubclones.get(inputNode.sample.sample)
    );

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

const slopeMultiplier = (vec: number[]) =>
  Math.abs(Math.sqrt(vec[0] ** 2 + vec[1] ** 2) / vec[0]);

const getTentacleOffset = (
  i: number,
  tentacleCount: number,
  tentacleSpacing: number,
  vec: number[] = [1, 0]
) => (i - tentacleCount / 2 + 0.5) * tentacleSpacing * slopeMultiplier(vec);

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

        // Make an aesthetically pleasing curve by adjusting the control point
        // positions based on the slope of the line between the input and output
        // points.
        const q = 2;
        const scDist =
          0.9 / ((slopeMultiplier([ix - ox, iy - oy]) + q) / (1 + q));

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
  passThroughSubclonesBySample: Map<SampleId, Set<Subclone>>,
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
      passThroughSubclonesBySample.get(sampleName),
      subcloneColors,
      coords.width,
      coords.height,
      layoutProps.tentacleWidth
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
  passThroughSubclones: Map<SampleId, Set<Subclone>>,
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
      passThroughSubclones,
      subcloneColors,
      layoutProps
    )
  );

  const tentacleBundles = collectTentacles(
    nodePlacement.keys(),
    shapersAndRegionsBySample,
    passThroughSubclones
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
