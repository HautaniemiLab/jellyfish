import { describe, it, expect } from "vitest";
import { findMissingColors } from "./tree.js";
import type { TreeNode } from "./tree.js";

// Concrete test node type
interface TestNode extends TreeNode<TestNode> {
  id: string;
}

function createNode(id: string): TestNode {
  return {
    id,
    parent: null,
    children: [],
  };
}

function addChild(parent: TestNode, child: TestNode) {
  child.parent = parent;
  parent.children.push(child);
}

// Helper to make a Set from an array (shorter to write in expectations)
function setOf<T>(...items: T[]): Set<T> {
  return new Set(items);
}

describe("findMissingColors", () => {
  it("returns empty sets when there are no colors at all", () => {
    const root = createNode("root");
    const child = createNode("child");
    addChild(root, child);

    const nodeColors = new Map<TestNode, Set<string>>();
    // No entries -> every node has zero colors

    const result = findMissingColors<string, TestNode>(root, nodeColors);

    expect(result.get(root)).toEqual(setOf());
    expect(result.get(child)).toEqual(setOf());
  });

  it("detects a simple missing color in a linear chain", () => {
    // root (red) -> mid () -> leaf (red)
    const root = createNode("root");
    const mid = createNode("mid");
    const leaf = createNode("leaf");
    addChild(root, mid);
    addChild(mid, leaf);

    const nodeColors = new Map<TestNode, Set<string>>([
      [root, setOf("red")],
      [mid, setOf()],
      [leaf, setOf("red")],
    ]);

    const result = findMissingColors<string, TestNode>(root, nodeColors);

    // root: red appears below but not above -> not missing
    expect(result.get(root)).toEqual(setOf());
    // mid: red above (root) and below (leaf), but not at mid -> missing red
    expect(result.get(mid)).toEqual(setOf("red"));
    // leaf: red above but not below -> not missing
    expect(result.get(leaf)).toEqual(setOf());
  });

  it("does not mark colors as missing when they are present on the node itself", () => {
    // root (red) -> mid (red) -> leaf (red)
    const root = createNode("root");
    const mid = createNode("mid");
    const leaf = createNode("leaf");
    addChild(root, mid);
    addChild(mid, leaf);

    const nodeColors = new Map<TestNode, Set<string>>([
      [root, setOf("red")],
      [mid, setOf("red")],
      [leaf, setOf("red")],
    ]);

    const result = findMissingColors<string, TestNode>(root, nodeColors);

    // No node should be missing red, because each has it
    expect(result.get(root)).toEqual(setOf());
    expect(result.get(mid)).toEqual(setOf());
    expect(result.get(leaf)).toEqual(setOf());
  });

  it("does not mark colors as missing when they appear only above or only below", () => {
    // Case 1: color only below
    // root () -> mid () -> leaf (red)
    const root1 = createNode("root1");
    const mid1 = createNode("mid1");
    const leaf1 = createNode("leaf1");
    addChild(root1, mid1);
    addChild(mid1, leaf1);

    const nodeColors1 = new Map<TestNode, Set<string>>([
      [root1, setOf()],
      [mid1, setOf()],
      [leaf1, setOf("red")],
    ]);

    const result1 = findMissingColors<string, TestNode>(root1, nodeColors1);

    expect(result1.get(root1)).toEqual(setOf());
    expect(result1.get(mid1)).toEqual(setOf());
    expect(result1.get(leaf1)).toEqual(setOf());

    // Case 2: color only above
    // root (red) -> mid () -> leaf ()
    const root2 = createNode("root2");
    const mid2 = createNode("mid2");
    const leaf2 = createNode("leaf2");
    addChild(root2, mid2);
    addChild(mid2, leaf2);

    const nodeColors2 = new Map<TestNode, Set<string>>([
      [root2, setOf("red")],
      [mid2, setOf()],
      [leaf2, setOf()],
    ]);

    const result2 = findMissingColors<string, TestNode>(root2, nodeColors2);

    expect(result2.get(root2)).toEqual(setOf());
    expect(result2.get(mid2)).toEqual(setOf());
    expect(result2.get(leaf2)).toEqual(setOf());
  });

  it("handles branching trees and marks missing colors on both branches", () => {
    //        root (red)
    //       /          \
    //    left ()      right ()
    //    /               \
    // leftLeaf(red)   rightLeaf(red)
    const root = createNode("root");
    const left = createNode("left");
    const right = createNode("right");
    const leftLeaf = createNode("leftLeaf");
    const rightLeaf = createNode("rightLeaf");

    addChild(root, left);
    addChild(root, right);
    addChild(left, leftLeaf);
    addChild(right, rightLeaf);

    const nodeColors = new Map<TestNode, Set<string>>([
      [root, setOf("red")],
      [left, setOf()],
      [right, setOf()],
      [leftLeaf, setOf("red")],
      [rightLeaf, setOf("red")],
    ]);

    const result = findMissingColors<string, TestNode>(root, nodeColors);

    // root has red itself -> not missing
    expect(result.get(root)).toEqual(setOf());
    // On both left and right, red appears above (root) and below (their leaf), but not on the branch node itself
    expect(result.get(left)).toEqual(setOf("red"));
    expect(result.get(right)).toEqual(setOf("red"));
    // Leaves: red above but not below -> not missing
    expect(result.get(leftLeaf)).toEqual(setOf());
    expect(result.get(rightLeaf)).toEqual(setOf());
  });

  it("handles multiple colors correctly", () => {
    //       root (A)
    //        |
    //      mid (B)
    //        |
    //     leaf (A, C)
    const root = createNode("root");
    const mid = createNode("mid");
    const leaf = createNode("leaf");
    addChild(root, mid);
    addChild(mid, leaf);

    const nodeColors = new Map<TestNode, Set<string>>([
      [root, setOf("A")],
      [mid, setOf("B")],
      [leaf, setOf("A", "C")],
    ]);

    const result = findMissingColors<string, TestNode>(root, nodeColors);

    // root: A appears below but not above -> not missing
    expect(result.get(root)).toEqual(setOf());
    // mid: A above (root) and below (leaf), but not at mid -> missing A
    // C appears only below; B appears only at mid
    expect(result.get(mid)).toEqual(setOf("A"));
    // leaf: A is above, but not below; C only here -> no missing colors
    expect(result.get(leaf)).toEqual(setOf());
  });

  it("works with non-string color types (e.g. numbers)", () => {
    // Simple chain with numeric colors
    const root = createNode("root");
    const mid = createNode("mid");
    const leaf = createNode("leaf");
    addChild(root, mid);
    addChild(mid, leaf);

    const nodeColors = new Map<TestNode, Set<number>>([
      [root, setOf(1)],
      [mid, setOf()],
      [leaf, setOf(1)],
    ]);

    const result = findMissingColors<number, TestNode>(root, nodeColors);

    expect(result.get(root)).toEqual(setOf());
    expect(result.get(mid)).toEqual(setOf(1));
    expect(result.get(leaf)).toEqual(setOf());
  });
});
