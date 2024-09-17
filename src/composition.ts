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
          rows.find((row) => row.subclone === subclone)?.clonalPrevalence ?? 0,
        ])
      ),
    (d) => d.sample
  );
}

/**
 * Clonal Prevalence and Cancer Cell Fraction are named and used similarly
 * as in the Supplementary Methods Figure 1 in paper "E-scape: interactive
 * visualization of single-cell phylogenetics and cancer evolution"
 * by Smith et al. (2017)
 */
export interface SubcloneMetrics {
  /**
   * The size of the subclone in the sample.
   */
  clonalPrevalence: number;

  /**
   * Size of the cluster, i.e. subclone's clonal prevalence plus the sum of the
   * cancer cell fractions of its immediate children.
   */
  cancerCellFraction: number;

  /**
   * Clonal prevalence of the subclone divided by its parent's cancer cell fraction.
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
    const clonalPrevalence = subclonalComposition.get(subclone) ?? 0;
    let cancerCellFraction = clonalPrevalence;

    for (const child of node.children) {
      cancerCellFraction += traverseClusterSize(child);
    }

    metricsMap.set(subclone, {
      clonalPrevalence,
      cancerCellFraction,
      fractionOfParent: 1,
    });

    return cancerCellFraction;
  }

  function traverseFractions(node: PhylogenyNode, parentCCF = 1) {
    const subclone = node.subclone;
    const metrics = metricsMap.get(subclone);
    metrics.fractionOfParent =
      parentCCF > 0 ? metrics.cancerCellFraction / parentCCF : 0;

    for (const child of node.children) {
      traverseFractions(child, metrics.cancerCellFraction);
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
      sum += metrics.clonalPrevalence * positionMap.get(subclone);
      totalSize += metrics.clonalPrevalence;
    }
    centresOfMass.set(sample, sum / totalSize);
  }

  return centresOfMass;
}
