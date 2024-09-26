import { SampleId, SampleRow } from "./data.js";
import { treeIterator, TreeNode, treeToNodeArray } from "./tree.js";

export const NODE_TYPES = {
  REAL_SAMPLE: "real",
  INFERRED_SAMPLE: "inferred",
  /** A pass for tentacles spanning multiple ranks */
  GAP: "gap",
};

export interface SampleTreeNode extends TreeNode<SampleTreeNode> {
  type: keyof typeof NODE_TYPES;
  rank: number;
  sample: (SampleRow & { indexNumber: number | null }) | null;
}

function samplesToNodes(samples: SampleRow[]): SampleTreeNode[] {
  // The inferred root that will be present in almost all cases
  const root = {
    sample: {
      sample: "root" as SampleId,
      displayName: "Inferred root",
      rank: 0,
      parent: null,
      indexNumber: null,
    },
    type: NODE_TYPES.INFERRED_SAMPLE,
    parent: null,
    children: [],
    rank: 0,
  } as SampleTreeNode;

  const nodes = samples.map(
    (sample, indexNumber) =>
      ({
        sample: { ...sample, indexNumber },
        type: NODE_TYPES.REAL_SAMPLE,
        parent: null,
        children: [],
        rank: sample.rank,
      } as SampleTreeNode)
  );

  validateRanks(nodes);

  return [root, ...nodes];
}

function validateRanks(nodes: SampleTreeNode[]) {
  if (
    nodes.some((node) => node.rank == null) &&
    nodes.some((node) => node.rank != null)
  ) {
    throw new Error("Rank must be either defined for all samples or none.");
  }

  // TODO: Allow user-defined root node
  if (nodes.some((node) => node.rank === 0)) {
    throw new Error("Rank 0 is reserved for the inferred root.");
  }
}

function createSampleTree(nodes: SampleTreeNode[]) {
  const root = nodes.find((node) => node.rank == 0);

  const nodeMap = new Map(nodes.map((node) => [node.sample.sample, node]));

  for (const node of nodes) {
    const parent = nodeMap.get(node.sample.parent);
    if (node.sample.parent && !parent) {
      throw new Error(`Parent "${node.sample.parent}" not found!`);
    }

    if (parent) {
      if (parent.rank >= node.rank) {
        throw new Error(
          `Parent "${parent.sample.sample}" has rank ${parent.rank} >= ${node.sample.sample}'s rank ${node.rank}`
        );
      }

      node.parent = parent;
      parent.children.push(node);
    }

    if (!node.parent && node != root) {
      node.parent = root;
      root.children.push(node);
    }
  }

  fixMissingRanks(root);

  return root;
}

function fixMissingRanks(sampleTree: SampleTreeNode) {
  for (const node of treeIterator(sampleTree)) {
    if (node.rank == null) {
      if (!node.parent) {
        throw new Error("Rank must be defined for the root node.");
      }
      node.rank = node.parent.rank + 1;
    }
  }
}

const findOccupiedRanks = (nodeArray: SampleTreeNode[]) =>
  Array.from(new Set(nodeArray.map((node) => node.rank)).values()).sort(
    (a, b) => a - b
  );

function addGaps(sampleTree: SampleTreeNode) {
  // Make the function pure by cloning the tree
  sampleTree = structuredClone(sampleTree);

  const nodes = treeToNodeArray(sampleTree);
  const occupiedRanks = findOccupiedRanks(nodes);

  for (const node of nodes) {
    const parent = node.parent;
    if (!parent) {
      continue;
    }

    // Ranks between the node and its parent
    const missingRanks = occupiedRanks.slice(
      occupiedRanks.indexOf(parent.rank) + 1,
      occupiedRanks.indexOf(node.rank)
    );

    // Add a gap for each missing rank
    let currentNode = parent;
    for (const rank of missingRanks) {
      const gap = {
        sample: null,
        type: NODE_TYPES.GAP,
        parent: currentNode,
        children: [node],
        rank,
      } as SampleTreeNode;

      node.parent = gap;
      currentNode.children[currentNode.children.findIndex((n) => n == node)] =
        gap;
      currentNode = gap;
    }
  }

  return sampleTree;
}

function occupiedRanksToPackMap(occupiedRanks: number[]) {
  return new Map(occupiedRanks.map((rank, i) => [rank, i]));
}

function packSampleTree(
  sampleTree: SampleTreeNode,
  packMap: Map<number, number>
) {
  sampleTree = structuredClone(sampleTree);

  for (const node of treeIterator(sampleTree)) {
    node.rank = packMap.get(node.rank);
  }

  return sampleTree;
}

export function createSampleTreeFromData(samples: SampleRow[]) {
  const nodes = samplesToNodes(samples);
  const sampleTree = createSampleTree(nodes);
  const rankPackMap = occupiedRanksToPackMap(
    findOccupiedRanks(treeToNodeArray(sampleTree))
  );

  const packedSampleTree = addGaps(packSampleTree(sampleTree, rankPackMap));

  return {
    sampleTree: packedSampleTree,
    rankPackMap,
  };
}
