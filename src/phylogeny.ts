import * as d3 from "d3";
import * as culori from "culori";
import { PhylogenyRow, Subclone } from "./data.js";
import { stratify, treeIterator, TreeNode, treeToNodeArray } from "./tree.js";

export interface PhylogenyNode extends TreeNode<PhylogenyNode> {
  subclone: Subclone;

  /**
   * Length of the branch leading to this node.
   */
  branchLength?: number;

  /**
   * The total length of the branches leading to this node.
   */
  totalBranchLength?: number;
}

export function buildPhylogenyTree(phylogenyTable: PhylogenyRow[]) {
  const tree = stratify(
    phylogenyTable,
    (d) => d.subclone,
    (d) => d.parent,
    (d) =>
      ({
        subclone: d.subclone,
        branchLength: d.branchLength,
        parent: null,
        children: [],
      } as PhylogenyNode)
  );

  for (const node of treeIterator(tree)) {
    node.totalBranchLength = node.branchLength ?? 0;
    if (node.parent) {
      node.totalBranchLength += node.parent.totalBranchLength;
    }
  }

  return tree;
}

export function rotatePhylogeny(
  phylogenyRoot: PhylogenyNode,
  subcloneRanks: Map<Subclone, number>
) {
  const tree = structuredClone(phylogenyRoot);

  // Ensure that the bells appear in the correct order
  const compareRank = (a: PhylogenyNode, b: PhylogenyNode) =>
    subcloneRanks.get(b.subclone) - subcloneRanks.get(a.subclone);

  // Place the least divergent subclones at the top
  const compareBranchLength = (a: PhylogenyNode, b: PhylogenyNode) =>
    a.totalBranchLength - b.totalBranchLength;

  // Composite comparator that first compares by rank and then by branch length.
  const compare = (a: PhylogenyNode, b: PhylogenyNode) => {
    const rankDiff = compareRank(a, b);
    return rankDiff != 0 ? rankDiff : compareBranchLength(a, b);
  };

  for (const node of treeIterator(tree)) {
    node.children = node.children.sort(compare);
  }

  return tree;
}

/**
 * Generates a phylogeny-aware color scheme. Nodes deeper in the phylogeny
 * will have darker and more saturated colors.
 */
export function generateColorScheme(
  phylogenyRoot: PhylogenyNode,
  hueOffset = 0
): Map<Subclone, string> {
  const colorScheme = new Map<Subclone, string>();

  const phylogenyArray = treeToNodeArray(phylogenyRoot);

  const maxTotalBranchLength = phylogenyArray.reduce(
    (max, node) => Math.max(max, node.totalBranchLength),
    0
  );

  const minBranchLength = phylogenyArray.reduce(
    (min, node) => Math.min(min, node.branchLength),
    Infinity
  );

  const domain = [
    phylogenyRoot.totalBranchLength,
    // Add minBranchLength to get some padding when the tree is shallow
    // and all the leaves have the same branch length. This avoids getting
    // an overly dark and saturated scheme.
    maxTotalBranchLength + minBranchLength,
  ];

  const lightnessScale = d3.scaleLinear().domain(domain).range([0.9, 0.68]);
  const chromaScale = d3.scaleLinear().domain(domain).range([0.02, 0.21]);

  let i = 0;

  // Pseudo-random but stable hue offset. The idea is to randomize the palette
  // between patients so that when there are only a few subclones, the colors
  // are still different. The rationale for color randomization is that the
  // different hues do not have any inherent meaning.
  hueOffset += (maxTotalBranchLength * 1000 * Math.PI) % 360;

  // Root is always gray and thus, needs no color from the color wheel.
  let n = phylogenyArray.length - 1;

  // However, leave a slight gap so that we can rotate the ugly browns away
  n += Math.max(1, n * 0.15);

  function traverse(node: PhylogenyNode) {
    const hue = ((i / n) * 360 + hueOffset) % 360;
    const chroma = i == 0 ? 0 : chromaScale(node.totalBranchLength);
    const lightness = lightnessScale(node.totalBranchLength);

    const lchColor = { mode: "oklch", l: lightness, c: chroma, h: hue };
    const rgbColor = culori.clampChroma(culori.rgb(lchColor));
    const hexColor = culori.formatHex(rgbColor);

    colorScheme.set(node.subclone, hexColor);

    i++;

    // The children with the smallest branch lengths get the smallest
    // difference in hue.
    const children = Array.from(node.children).sort(
      (a, b) => a.branchLength - b.branchLength
    );

    for (const child of children) {
      traverse(child);
    }
  }

  traverse(phylogenyRoot);

  return colorScheme;
}
