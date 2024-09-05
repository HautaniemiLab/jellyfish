import * as d3 from "d3";
import { CompositionRow, Subclone } from "./data.js";
import { PhylogenyNode } from "./phylogeny.js";

export function getProportionsBySamples(compositionsTable: CompositionRow[]) {
  const subclones = new Set(compositionsTable.map((d) => d.subclone));

  // Return a Map of Maps, first level has the sample, second has the subclone.
  return new Map(
    [...d3.group(compositionsTable, (d) => d.sample)].map(([sample, rows]) => {
      const subcloneMap = new Map(
        rows.map((row) => [row.subclone, row.proportion])
      );
      // With all (including the missing) subclones
      const completedMap = new Map(
        [...subclones.values()].map((subclone) => [
          subclone,
          subcloneMap.get(subclone) ?? 0,
        ])
      );
      return [sample, completedMap];
    })
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
    metrics.fractionOfParent = metrics.clusterSize / parentClusterSize;

    for (const child of node.children) {
      traverseFractions(child, metrics.clusterSize);
    }
  }

  traverseClusterSize(phylogenyRoot);
  traverseFractions(phylogenyRoot);

  return metricsMap;
}
