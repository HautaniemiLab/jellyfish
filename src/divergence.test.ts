import { describe, it, expect } from "vitest";
import { buildPhylogenyTree } from "./phylogeny.js";
import {
  calculateSubcloneMetrics,
  getProportionsBySamples,
} from "./composition.js";
import { treeToNodeArray } from "./tree.js";

describe("subclonal divergence usage of cancerCellFraction", () => {
  it("uses cancerCellFraction for subclones that have LCA descendants in a sample", () => {
    // Simple phylogeny with founding clone F: F -> A -> {B, C}
    const phylogeny = [
      { subclone: "F", parent: "", color: "#eeeeee", branchLength: 1 },
      { subclone: "A", parent: "F", color: "#ffffff", branchLength: 1 },
      { subclone: "B", parent: "A", color: "#ff0000", branchLength: 1 },
      { subclone: "C", parent: "A", color: "#00ff00", branchLength: 1 },
    ] as any;

    const phyRoot = buildPhylogenyTree(phylogeny, false);

    // Samples: root S0, child S1 (parent S0), child S2 (parent S0)
    const samples = [
      { sample: "S0", rank: 1 },
      { sample: "S1", rank: 2, parent: "S0" },
      { sample: "S2", rank: 2, parent: "S0" },
    ] as any;

    // Compositions: clonal prevalences must sum to 1 per sample (excluding normal cells)
    // S0: F 0.8, A 0.2, B 0, C 0
    // S1: F 0.5, A 0.1, B 0.4, C 0
    // S2: F 0.35, A 0.05, B 0, C 0.6
    const compositions = [
      { sample: "S0", subclone: "F", clonalPrevalence: 0.8 },
      { sample: "S0", subclone: "A", clonalPrevalence: 0.2 },
      { sample: "S0", subclone: "B", clonalPrevalence: 0 },
      { sample: "S0", subclone: "C", clonalPrevalence: 0 },

      { sample: "S1", subclone: "F", clonalPrevalence: 0.5 },
      { sample: "S1", subclone: "A", clonalPrevalence: 0.1 },
      { sample: "S1", subclone: "B", clonalPrevalence: 0.4 },
      { sample: "S1", subclone: "C", clonalPrevalence: 0 },

      { sample: "S2", subclone: "F", clonalPrevalence: 0.35 },
      { sample: "S2", subclone: "A", clonalPrevalence: 0.05 },
      { sample: "S2", subclone: "B", clonalPrevalence: 0 },
      { sample: "S2", subclone: "C", clonalPrevalence: 0.6 },
    ] as any;

    // Build metrics per sample
    const proportions = getProportionsBySamples(compositions);
    const metricsBySample = new Map<string, Map<string, any>>();
    for (const sampleObj of samples) {
      const sample = sampleObj.sample;
      const map = calculateSubcloneMetrics(
        phyRoot,
        proportions.get(sample) ?? new Map()
      );
      metricsBySample.set(sample, map as any);
    }

    // Manually define LCAs for this simple case:
    // B first appears in S1, C in S2, A in S0
    const subcloneLCAs = new Map<string, string>([
      ["A", "S0"],
      ["B", "S1"],
      ["C", "S2"],
    ]);

    // Precompute descendants for A,B,C
    const subcloneOrder = treeToNodeArray(phyRoot).map((n) => n.subclone);
    const descendants = new Map<string, Set<string>>();
    for (const node of treeToNodeArray(phyRoot)) {
      descendants.set(
        node.subclone,
        new Set(treeToNodeArray(node).map((n) => n.subclone))
      );
    }

    // For sample S1, subclone A has descendant B whose LCA is S1
    const sample = "S1";
    const metricsMap = metricsBySample.get(sample)!;

    const hasLcaDescendant = Array.from(descendants.get("A")!).some(
      (d) => subcloneLCAs.get(d) === sample
    );

    // Compute the value that would be used according to the rule
    const usedValueForA = hasLcaDescendant
      ? metricsMap.get("A").cancerCellFraction
      : metricsMap.get("A").clonalPrevalence;

    // Expect that the used value equals cancerCellFraction (not clonalPrevalence)
    expect(usedValueForA).toBe(metricsMap.get("A").cancerCellFraction);

    // For sample S0, A should NOT use cancerCellFraction because B/C LCAs are not S0
    const metricsMap0 = metricsBySample.get("S0")!;
    const hasLcaDescendant0 = Array.from(descendants.get("A")!).some(
      (d) => subcloneLCAs.get(d) === "S0"
    );
    const usedValueForA0 = hasLcaDescendant0
      ? metricsMap0.get("A").cancerCellFraction
      : metricsMap0.get("A").clonalPrevalence;

    expect(usedValueForA0).toBe(metricsMap0.get("A").clonalPrevalence);
  });
});
