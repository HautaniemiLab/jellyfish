export interface TreeNode<T extends TreeNode<T>> {
  parent: T | null;
  children: T[];
}

export function treeToNodeArray<T extends TreeNode<T>>(root: T): T[] {
  const nodes: T[] = [];
  function process(node: T) {
    nodes.push(node);
    for (const child of node.children) {
      process(child);
    }
  }
  process(root);
  return nodes;
}

/**
 * Iterate over the tree in a depth-first manner.
 */
export function* treeIterator<T extends TreeNode<T>>(root: T): Generator<T> {
  if (!root) {
    throw new Error("Trying to iterate over a null or undefined tree node!");
  }
  yield root;
  for (const child of root.children) {
    yield* treeIterator(child);
  }
}

export function stratify<T, N extends TreeNode<N>>(
  data: T[],
  id: (d: T) => string,
  parentId: (d: T) => string,
  nodeBuilder: (d: T) => N
): N {
  const nodes = new Map<string, N>();
  for (const d of data) {
    const node = nodeBuilder(d);
    nodes.set(id(d), node);
  }
  for (const d of data) {
    const parent = nodes.get(parentId(d));
    const node = nodes.get(id(d));
    if (parent) {
      node.parent = parent;
      parent.children.push(node);
    }
  }

  for (const node of nodes.values()) {
    if (node.parent == null) {
      return node;
    }
  }
}

/**
 * Computes, for each node, the set of "missing" colors:
 * colors that appear somewhere above and somewhere below the node
 * on some root–leaf path, but are not present in the node itself.
 *
 * Created with the help of ChatGPT 5.1.
 */
export function findMissingColors<
  T, // color type (e.g., Subclone)
  N extends TreeNode<N> // node type
>(root: N, nodeColors: Map<N, Set<T>>): Map<N, Set<T>> {
  // Colors that appear strictly in descendants of each node
  const downColors = new Map<N, Set<T>>();
  // Final result: missing colors per node
  const missingPerNode = new Map<N, Set<T>>();

  const emptySet = new Set<T>();

  const getColors = (node: N): Set<T> => nodeColors.get(node) ?? emptySet;

  // ---- Pass 1: bottom-up, fill `downColors` ----
  function dfsDown(node: N): Set<T> {
    const acc = new Set<T>();

    for (const child of node.children) {
      const childDown = dfsDown(child);
      const childColors = getColors(child);

      for (const c of childColors) {
        acc.add(c);
      }
      // colors in descendants of child
      for (const c of childDown) {
        acc.add(c);
      }
    }

    downColors.set(node, acc);
    return acc;
  }

  dfsDown(root);

  // ---- Pass 2: top-down, compute missing colors ----
  function dfsUp(node: N, upSet: Set<T>): void {
    const nodeDown = downColors.get(node) ?? emptySet;
    const nodeOwnColors = getColors(node);
    const nodeMissing = new Set<T>();

    for (const color of upSet) {
      if (nodeDown.has(color) && !nodeOwnColors.has(color)) {
        nodeMissing.add(color);
      }
    }

    missingPerNode.set(node, nodeMissing);

    // Extend the up-set for children: path now includes this node's colors
    const newUp = new Set(upSet);
    for (const c of nodeOwnColors) {
      newUp.add(c);
    }

    for (const child of node.children) {
      dfsUp(child, newUp);
    }
  }

  dfsUp(root, new Set<T>());

  return missingPerNode;
}
