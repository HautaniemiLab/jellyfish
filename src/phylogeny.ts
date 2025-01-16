import * as d3 from "d3";
import * as culori from "culori";
import { PhylogenyRow, Subclone } from "./data.js";
import { stratify, treeIterator, TreeNode, treeToNodeArray } from "./tree.js";
import { SeededRNG } from "./utils.js";

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
 *
 * @param phylogenyRoot Root of the phylogeny tree
 * @param hueOffset Offset in the hue of the colors
 * @param normalRoot Whether the root of the phylogeny represents non-aberrant cells
 */
export function generateColorScheme(
  phylogenyRoot: PhylogenyNode,
  hueOffset = 0,
  normalRoot: boolean
): Map<Subclone, string> {
  const originalPhylogenyRoot = phylogenyRoot;

  console.log(normalRoot);
  // If the root is a normal sample and has only one child (the founding clone),
  // use the founding clone as the root. The original root represents the normal
  // cells.
  if (normalRoot && phylogenyRoot.children.length == 1) {
    phylogenyRoot = phylogenyRoot.children[0];
  }

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
  const chromaScale = d3.scaleLinear().domain(domain).range([0.025, 0.21]);

  // Root is always gray and thus, needs no color from the color wheel.
  let n = phylogenyArray.length - 1;

  // However, leave a slight gap so that we can rotate the ugly browns away
  n += Math.max(1, n * 0.15);

  function getColors(hueOffset: number) {
    const colors = new Map<Subclone, string>();

    if (
      normalRoot &&
      (phylogenyRoot != originalPhylogenyRoot ||
        originalPhylogenyRoot.children.length > 1)
    ) {
      // TODO: Configurable color for the normal root
      colors.set(originalPhylogenyRoot.subclone, "#ffffff");
    }

    let i = 0;
    function traverse(node: PhylogenyNode) {
      const hue = ((i / n) * 360 + hueOffset) % 360;
      const chroma = i == 0 ? 0 : chromaScale(node.totalBranchLength);
      const lightness = lightnessScale(node.totalBranchLength);

      const lchColor = { mode: "oklch", l: lightness, c: chroma, h: hue };
      const rgbColor = culori.clampChroma(culori.rgb(lchColor));
      const hexColor = culori.formatHex(rgbColor);

      if (!colors.has(node.subclone)) {
        // Don't overwrite the normal color
        colors.set(node.subclone, hexColor);
      }

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
    return colors;
  }

  // Find such a hue offset that the colors are as far away from the ugly
  // colors as possible.
  // Using randomization here as we want some variation in the colors.
  const rng = SeededRNG(maxTotalBranchLength);
  let bestDistance = 0;
  let bestOffset = 0;
  const diff = culori.differenceEuclidean("oklch");
  for (let i = 0; i < 20; i++) {
    const offset = rng() * 360;
    const distance = Array.from(getColors(offset).values())
      .map((color) => uglyColors.map((uglyColor) => diff(color, uglyColor)))
      .flat()
      .reduce((a, b) => Math.min(a, b), Infinity);
    if (distance > bestDistance) {
      bestDistance = distance;
      bestOffset = offset;
    }

    // As we are randomizing, we allow some variation when the result
    // is good enough. Just to not always get the same scheme, particularly
    // when there are only a few subclones.
    if (bestDistance > 0.08) {
      break;
    }
  }

  // Get the final color scheme based on the least ugliest
  // hue offset + the user provided hue offset.
  return getColors(bestOffset + hueOffset);
}

/**
 * Dirty brown (dark yellow) colors that should be avoided in the color scheme.
 */
const uglyColors = [
  "#b99f00",
  "#c09900",
  "#c3a500",
  "#c39100",
  "#b8ad00",
  "#cab547",
  "#9a8200",
].map((color) => culori.oklch(color));
