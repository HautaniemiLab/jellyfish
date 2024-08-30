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

export function* treeIterator<T extends TreeNode<T>>(root: T): Generator<T> {
  yield root;
  for (const child of root.children) {
    yield* treeIterator(child);
  }
}
