import { SampleRow } from "./data.js";
import { TreeNode, treeToNodeArray } from "./tree.js";

export const NODE_TYPES = {
  REAL_SAMPLE: "real",
  INFERRED_SAMPLE: "inferred",
  /** A pass for tentacles spanning multiple ranks */
  GAP: "gap",
};

export type RankMap = Map<string, number>;

export interface SampleTreeNode extends TreeNode<SampleTreeNode> {
  type: keyof typeof NODE_TYPES;
  rank: number;
  sample: any | null;
}

function samplesToNodes(
  samples: SampleRow[],
  rankMap: RankMap
): SampleTreeNode[] {
  // The inferred root that will be present in almost all cases
  const root = {
    sample: {
      sample: "root",
      displayName: "Inferred root",
      size: null,
      timepoint: null,
    },
    type: NODE_TYPES.INFERRED_SAMPLE,
    parent: null,
    children: [],
    rank: 0,
  } as SampleTreeNode;

  function getRankOrThrow(timepoint: string) {
    const rank = rankMap.get(timepoint);
    if (!rank) {
      throw new Error("Cannot find a rank for the timepoint: " + timepoint);
    }
    return rank;
  }

  const nodes = [
    root,
    ...samples.map(
      (sample) =>
        ({
          sample,
          type: NODE_TYPES.REAL_SAMPLE,
          parent: null,
          children: [],
          rank: getRankOrThrow(sample.timepoint),
        } as SampleTreeNode)
    ),
  ];

  return nodes;
}

function createSampleTree(nodes: SampleTreeNode[]) {
  const maxRank = nodes
    .map((node) => node.rank)
    .reduce((a, b) => Math.max(a, b), 0);

  // A map for convenient lookup
  const nodesByRank = new Map();
  for (let rank = 0; rank <= maxRank; rank++) {
    nodesByRank.set(rank, []);
  }

  for (const node of nodes) {
    nodesByRank.get(node.rank).push(node);
  }

  const root = nodes.find((node) => node.rank == 0);

  for (const node of nodes) {
    for (let rank = node.rank - 1; rank >= 0; rank--) {
      // Samples that have the same site
      const candidateNodes = nodesByRank
        .get(rank)
        .filter((candidate) => candidate.sample.site == node.sample.site);

      // Check if an earlier rank has exactly one sample with the same site
      if (candidateNodes.length == 1) {
        // If there is, use that as the parent
        const parent = candidateNodes[0];

        node.parent = parent;
        parent.children.push(node);
        break;
      }
    }

    // No sample with the same site found. Connect to the inferred root.
    if (!node.parent && node.rank > 0) {
      node.parent = root;
      root.children.push(node);
    }
  }

  return root;
}

const findOccupiedRanks = (nodeArray) =>
  [...new Set(nodeArray.map((node) => node.rank)).values()].sort(
    (a, b) => a - b
  );

function addGaps(sampleTree: SampleTreeNode) {
  // Make the function pure by cloning the tree
  sampleTree = structuredClone(sampleTree);

  const nodes = treeToNodeArray(sampleTree);
  const occupiedRanks = findOccupiedRanks(nodes);

  const rankPredecessors = new Map(
    occupiedRanks.map((rank, i) => [rank, occupiedRanks[i - 1]])
  );

  for (const node of nodes) {
    const parent = node.parent;
    if (!parent) {
      continue;
    }

    // Predecessor ranks
    const current = parent.rank;
    const expected = rankPredecessors.get(node.rank);

    if (expected != null && current != expected) {
      // Add a gap between the node and its parent
      const gap = {
        sample: null,
        type: NODE_TYPES.GAP,
        parent,
        children: [node],
        rank: expected,
      } as SampleTreeNode;

      node.parent = gap;
      const index = parent.children.findIndex((n) => n == node);
      if (index < 0) {
        throw new Error("Corrupted tree!");
      }
      parent.children[index] = gap;
    }
  }

  return sampleTree;
}

function rankTableToRankMap(rankTable) {
  return new Map(rankTable.map((d) => [d.timepoint, +d.rank]));
}

function occupiedRanksToPackMap(occupiedRanks, rankMap) {
  // Map rank numbers to timepoints
  const reverseRankMap = new Map(
    [...rankMap.entries()].map(([k, v]) => [v, k])
  );

  return new Map(
    occupiedRanks.map((rank, i) => [
      rank,
      { rank: i, timepoint: reverseRankMap.get(rank) },
    ])
  );
}

function packMapToPackedRankMap(packMap) {
  return new Map([...packMap.values()].map((d) => [d.rank, d.timepoint]));
}

function packSampleTree(sampleTree, packMap) {
  sampleTree = structuredClone(sampleTree);

  for (const node of treeToNodeArray(sampleTree)) {
    node.rank = packMap.get(node.rank).rank;
  }

  return sampleTree;
}
