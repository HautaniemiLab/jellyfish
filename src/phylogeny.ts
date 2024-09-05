import { PhylogenyRow, Subclone } from "./data.js";
import { stratify, treeIterator, TreeNode } from "./tree.js";

export interface PhylogenyNode extends TreeNode<PhylogenyNode> {
  subclone: Subclone;
}

export function buildPhylogenyTree(phylogenyTable: PhylogenyRow[]) {
  return stratify(
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
}

export function rotatePhylogeny(
  phylogenyRoot: PhylogenyNode,
  subcloneRanks: Map<Subclone, number>
) {
  const tree = structuredClone(phylogenyRoot);

  const compare = (a: PhylogenyNode, b: PhylogenyNode) => {
    return subcloneRanks.get(b.subclone) - subcloneRanks.get(a.subclone);
  };

  for (const node of treeIterator(tree)) {
    node.children = node.children.sort(compare);
  }

  return tree;
}
