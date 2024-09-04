import { PhylogenyRow, Subclone } from "./data.js";
import { stratify, TreeNode } from "./tree.js";

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
