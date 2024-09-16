import * as d3 from "d3";
import { CompositionRow, SampleId, Subclone } from "./data.js";
import { PhylogenyNode } from "./phylogeny.js";
import { treeToNodeArray } from "./tree.js";

export function getProportionsBySamples(
  compositionsTable: CompositionRow[]
): Map<SampleId, Map<Subclone, number>> {
  const subclones = Array.from(
    new Set(compositionsTable.map((d) => d.subclone))
  );

  return d3.rollup(
    compositionsTable,
    (rows) =>
      new Map(
        subclones.map((subclone) => [
          subclone,
          rows.find((row) => row.subclone === subclone)?.proportion ?? 0,
        ])
      ),
    (d) => d.sample
  );
}

export interface SubcloneMetrics {
  /**
   * The size (VAF, Fraction, Proportien, whatever) of the subclone in the sample.
   */
  subcloneSize: number;

  /**
   * Size of the cluster, i.e. the sum of the size of the subclone and all its descendants.
   */
  clusterSize: number;

  /**
   * The subclone size scaled to its parent cluster's size.
   */
  fractionOfParent: number;
}

export type SubcloneMetricsMap = Map<Subclone, SubcloneMetrics>;

export function calculateSubcloneMetrics(
  phylogenyRoot: PhylogenyNode,
  subclonalComposition: Map<Subclone, number>
): SubcloneMetricsMap {
  const metricsMap = new Map<Subclone, SubcloneMetrics>();

  function traverseClusterSize(node: PhylogenyNode) {
    const subclone = node.subclone;
    const subcloneSize = subclonalComposition.get(subclone) ?? 0;
    let clusterSize = subcloneSize;

    for (const child of node.children) {
      clusterSize += traverseClusterSize(child);
    }

    metricsMap.set(subclone, {
      subcloneSize,
      clusterSize,
      fractionOfParent: 1,
    });

    return clusterSize;
  }

  function traverseFractions(node: PhylogenyNode, parentClusterSize = 1) {
    const subclone = node.subclone;
    const metrics = metricsMap.get(subclone);
    metrics.fractionOfParent =
      parentClusterSize > 0 ? metrics.clusterSize / parentClusterSize : 0;

    for (const child of node.children) {
      traverseFractions(child, metrics.clusterSize);
    }
  }

  traverseClusterSize(phylogenyRoot);
  traverseFractions(phylogenyRoot);

  return metricsMap;
}

/**
 * Centers of mass are calculated as the weighted average of the positions
 * of the subclones in the phylogeny. This is a simple way to determine
 * a preferred order of the samples within each column.
 */
export function calculateCentresOfMass(
  phylogenyRoot: PhylogenyNode,
  metricsBySample: Map<SampleId, SubcloneMetricsMap>
): Map<SampleId, number> {
  const centresOfMass = new Map<SampleId, number>();

  // This could be improved a bit for non-balanced trees
  const phylogenyNodes = treeToNodeArray(phylogenyRoot);
  const positionMap = new Map(
    phylogenyNodes.map((node, index) => [
      node.subclone,
      index / phylogenyNodes.length,
    ])
  );

  for (const [sample, metricsMap] of metricsBySample) {
    let sum = 0;
    let totalSize = 0;
    for (const [subclone, metrics] of metricsMap) {
      sum += metrics.subcloneSize * positionMap.get(subclone);
      totalSize += metrics.subcloneSize;
    }
    centresOfMass.set(sample, sum / totalSize);
  }

  return centresOfMass;
}
