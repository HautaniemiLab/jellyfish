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
