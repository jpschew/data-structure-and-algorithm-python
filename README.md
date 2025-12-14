# Data Structures and Algorithms (DSA)

A Python implementation of fundamental data structures with comprehensive test cases.

## Table of Contents

- [Project Structure](#project-structure)
- [Data Structures](#data-structures)
  - [1. Linked Lists](#1-linked-lists)
    - [1.1 Head Only vs Head + Tail Reference](#11-head-only-vs-head--tail-reference)
    - [1.2 Singly vs Doubly Linked List](#12-singly-vs-doubly-linked-list)
  - [2. Stack](#2-stack)
  - [3. Queue](#3-queue)
    - [3.1 Queue Using Two Stacks](#31-queue-using-two-stacks)
  - [4. Priority Queue](#4-priority-queue)
  - [5. Trees](#5-trees)
    - [5.1 Binary Tree](#51-binary-tree)
    - [5.2 Binary Search Tree](#52-binary-search-tree)
    - [5.3 AVL Tree](#53-avl-tree-self-balancing-bst)
  - [6. Heaps](#6-heaps)
    - [6.1 PriorityQueueHeap](#61-priorityqueueheap)
  - [7. Tries](#7-tries)
    - [7.1 Trie Operations](#71-trie-operations)
    - [7.2 Implementation Comparison](#72-implementation-comparison)
  - [8. Graphs](#8-graphs)
    - [8.1 Graph Operations](#81-graph-operations)
    - [8.2 Topological Sort](#82-topological-sort)
    - [8.3 Shortest Path (Dijkstra's Algorithm)](#83-shortest-path-dijkstras-algorithm)
    - [8.4 Minimum Spanning Tree (Prim's Algorithm)](#84-minimum-spanning-tree-prims-algorithm)
    - [8.5 Implementation Comparison](#85-implementation-comparison)
    - [8.6 Why No Node/Edge Classes?](#86-why-no-nodeedge-classes)
- [Running Tests](#running-tests)
- [Requirements](#requirements)

## Project Structure

```
dsa/
└── src/
    ├── main.py                 # Test runner
    ├── linked_list/
    │   └── linked_list.py      # Linked list implementations
    ├── stack_queue/
    │   ├── stack.py            # Stack implementation
    │   └── queue.py            # Queue implementation
    ├── trees/
    │   ├── binary_tree.py      # Binary Search Tree implementation
    │   └── avl_tree.py         # AVL Tree (self-balancing BST)
    ├── heaps/
    │   └── binary_heap.py      # MinHeap and MaxHeap implementations
    ├── tries/
    │   └── trie.py             # Trie implementations (4 variants)
    └── graphs/
        └── graph.py            # Graph implementations (Matrix and List)
```

## Data Structures

### 1. Linked Lists

Three variants with different trade-offs:

| Class | Description | Key Feature |
|-------|-------------|-------------|
| `LinkedListHeadOnly` | Singly linked with head only | Minimal memory |
| `LinkedListHeadTail` | Singly linked with head and tail | O(1) append |
| `DoublyLinkedList` | Doubly linked with head and tail | O(1) delete_last, bidirectional traversal |

**Time Complexity Comparison:**

| Operation | Head Only | Head+Tail | Doubly Linked |
|-----------|-----------|-----------|---------------|
| prepend() | O(1) | O(1) | O(1) |
| append() | O(n) | O(1) | O(1) |
| insert(data, index) | O(n) | O(n) | O(n) |
| delete(value) | O(n) | O(n) | O(n) |
| delete_first() | O(1) | O(1) | O(1) |
| delete_last() | O(n) | O(n) | O(1) |
| delete_at(index) | O(n) | O(n) | O(n) |
| search() | O(n) | O(n) | O(n) |

#### 1.1 Head Only vs Head + Tail Reference

```
Head Only:                      Head + Tail:

HEAD                            HEAD                 TAIL
  |                               |                    |
  v                               v                    v
[1] -> [2] -> [3] -> None       [1] -> [2] -> [3] -> None
```

| Aspect | Head Only | Head + Tail |
|--------|-----------|-------------|
| Memory | Less (1 pointer) | More (2 pointers) |
| append() | O(n) - must traverse | O(1) - use tail |
| prepend() | O(1) | O(1) |
| delete_last() | O(n) | O(n)* |
| Implementation | Simpler | Slightly more complex |

*delete_last() is still O(n) for singly linked with tail because we need the node before tail. Only doubly linked achieves O(1).

**When to Use Which:**

| Scenario | Use | Why |
|----------|-----|-----|
| Stack (LIFO) | Head Only | Only need prepend/delete_first - both O(1) |
| Queue (FIFO) | Head + Tail | Need O(1) append (enqueue) and O(1) delete_first (dequeue) |
| Frequent append operations | Head + Tail | O(1) vs O(n) append |
| Minimal memory usage | Head Only | One less pointer to maintain |
| Simple implementation | Head Only | No tail pointer to update |

**Summary:**
- **Head Only**: Best for stack operations or when memory is critical
- **Head + Tail**: Best for queue operations or frequent appends

#### 1.2 Singly vs Doubly Linked List

```
Singly Linked:     Doubly Linked:

  +------+         +------+------+
  | data |         | prev | data |
  | next |---->    |<-----| next |----->
  +------+         +------+------+

  Each node: 2 fields    Each node: 3 fields
  (data + next)          (prev + data + next)
```

| Aspect | Singly Linked | Doubly Linked |
|--------|---------------|---------------|
| Memory per node | Less (no prev pointer) | More (extra prev pointer) |
| Traversal | Forward only | Forward and backward |
| Delete node (given pointer) | O(n) - need to find previous | O(1) - has prev pointer |
| Delete last | O(n) - must traverse | O(1) - use tail.prev |
| Implementation | Simpler | More complex |

**When to Use Which:**

| Scenario | Use | Why |
|----------|-----|-----|
| Memory constrained | Singly | Less memory per node |
| Simple stack/queue | Singly | Only need head/tail operations |
| Frequent delete_last() | Doubly | O(1) vs O(n) |
| Need backward traversal | Doubly | Has prev pointer |
| Browser history (back/forward) | Doubly | Navigate both directions |
| Undo/Redo functionality | Doubly | Move back and forth |
| LRU Cache | Doubly | O(1) removal from middle |
| Music playlist (prev/next) | Doubly | Navigate both directions |

**Summary:**
- **Default choice**: Singly linked (simpler, less memory)
- **Choose Doubly when**: Need O(1) delete_last, backward traversal, or O(1) removal of arbitrary nodes

**Usage:**
```python
from linked_list.linked_list import LinkedListHeadOnly, LinkedListHeadTail, DoublyLinkedList

# Singly linked list
ll = LinkedListHeadTail()
ll.append(1)
ll.append(2)
ll.prepend(0)
print(ll)  # 0 -> 1 -> 2

# Insert at specific index
ll.insert(99, 2)       # 0 -> 1 -> 99 -> 2

# Delete operations
ll.delete_first()      # Returns 0,  list: 1 -> 99 -> 2
ll.delete_last()       # Returns 2,  list: 1 -> 99
ll.delete_at(1)        # Returns 99, list: 1

# Doubly linked list
dll = DoublyLinkedList()
dll.append(1)
dll.append(2)
dll.append(3)
print(dll)                      # 1 <-> 2 <-> 3
print(dll.reverse_traverse())   # [3, 2, 1]
dll.delete_last()               # O(1) - returns 3
```

### 2. Stack

LIFO (Last In, First Out) data structure.

| Operation | Description | Time |
|-----------|-------------|------|
| push(data) | Add to top | O(1) |
| pop() | Remove from top | O(1) |
| peek() | View top | O(1) |
| is_empty() | Check if empty | O(1) |

**Usage:**
```python
from stack_queue.stack import Stack

stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())   # 3
print(stack.peek())  # 2
```

### 3. Queue

FIFO (First In, First Out) data structure.

| Operation | Description | Time |
|-----------|-------------|------|
| enqueue(data) | Add to rear | O(1) |
| dequeue() | Remove from front | O(1) |
| peek() | View front | O(1) |
| is_empty() | Check if empty | O(1) |

**Usage:**
```python
from stack_queue.queue import Queue

queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.dequeue())  # 1
print(queue.peek())     # 2
```

#### 3.1 Queue Using Two Stacks

Alternative queue implementation using two stacks with amortized O(1) operations.

| Operation | Description | Time (amortized) | Time (worst) |
|-----------|-------------|------------------|--------------|
| enqueue(data) | Add to rear | O(1) | O(1) |
| dequeue() | Remove from front | O(1) | O(n) |
| peek() | View front | O(1) | O(n) |

```python
from stack_queue.queue import QueueUsingStacks

queue = QueueUsingStacks()
queue.enqueue(1)
queue.enqueue(2)
print(queue.dequeue())  # 1
```

### 4. Priority Queue

Min-Priority Queue implementations (lower value = higher priority). Five variants comparing different approaches:

| Implementation | enqueue() | dequeue() | peek() | Location |
|----------------|-----------|-----------|--------|----------|
| `PriorityQueueUnsortedList` | O(1) | O(n) | O(n) | queue.py |
| `PriorityQueueSortedList` | O(n) | O(1) | O(1) | queue.py |
| `PriorityQueueUnsortedLinkedList` | O(1) | O(n) | O(n) | queue.py |
| `PriorityQueueSortedLinkedList` | O(n) | O(1) | O(1) | queue.py |
| `PriorityQueueHeap` | O(log n) | O(log n) | O(1) | binary_heap.py |

*See [6.1 PriorityQueueHeap](#61-priorityqueueheap) for detailed heap-based implementation.*

**Trade-offs:**
- **Unsorted**: Fast insert, slow removal (good for many inserts, few removals)
- **Sorted**: Slow insert, fast removal (good for few inserts, many removals)
- **Heap**: Balanced performance for both (best for mixed operations)

**Implementation Difference:**

| Operation | Unsorted | Sorted |
|-----------|----------|--------|
| enqueue() | Just append to end | Find correct position + shift elements |
| dequeue() | Scan all elements to find min | Min is always at end, just pop |
| peek() | Scan all elements to find min | Min is always at end, just look |

*Note: `peek()` and `dequeue()` have the same complexity because `peek()` looks at the highest priority element, not just the front of the queue. Both must find the minimum first.*

The O(n) cost is paid either on insert (sorted) or on removal (unsorted).

**How They Work:**

*Note: Priority queue removes by priority, not FIFO like regular queue.*

**List Implementation:**
```
UNSORTED LIST: Insert anywhere, search when removing
enqueue("Low", 3)     → [(3,Low)]
enqueue("High", 1)    → [(3,Low), (1,High)]
enqueue("Urgent", 0)  → [(3,Low), (1,High), (0,Urgent)]
dequeue()             → scan all, find min(0) → returns "Urgent"

SORTED LIST: Store descending (min at END), pop from end
enqueue("Low", 3)     → [(3,Low)]
enqueue("High", 1)    → [(3,Low), (1,High)]         ← sorted position
enqueue("Urgent", 0)  → [(3,Low), (1,High), (0,Urgent)]  ← min at end
dequeue()             → pop() from end → returns "Urgent" (O(1))
```

**Linked List Implementation:**
```
UNSORTED LINKED LIST: Insert at head, search when removing
enqueue("Low", 3)     → (3,Low)
enqueue("High", 1)    → (1,High) → (3,Low)
enqueue("Urgent", 0)  → (0,Urgent) → (1,High) → (3,Low)
dequeue()             → scan all, find min(0) → returns "Urgent"

SORTED LINKED LIST: Store ascending (min at HEAD), remove from head
enqueue("Low", 3)     → (3,Low)
enqueue("High", 1)    → (1,High) → (3,Low)          ← sorted position
enqueue("Urgent", 0)  → (0,Urgent) → (1,High) → (3,Low)  ← min at head
dequeue()             → remove head → returns "Urgent" (O(1))
```

**Where do unsorted implementations insert?**

| Data Structure | Insert At | Why |
|----------------|-----------|-----|
| List | End (append) | `list.append()` is O(1) |
| Linked List | Head (prepend) | Head insertion is O(1), tail is O(n) |

**Why different remove positions?**

| Data Structure | Sorted Order | Remove From | Why |
|----------------|--------------|-------------|-----|
| List | Descending (min at end) | End | `pop()` is O(1), `pop(0)` is O(n) |
| Linked List | Ascending (min at head) | Head | Head removal is O(1), tail is O(n) |

**Why TAIL reference doesn't help?**

| Implementation | Bottleneck | TAIL helps? |
|----------------|------------|-------------|
| Unsorted | O(n) scan to find min | No - still need to scan all |
| Sorted | O(n) traverse to find position | No - still need to traverse |

TAIL helps for O(1) append, but priority queue bottleneck is **finding**, not inserting.

**Why Doubly Linked List (HEAD + TAIL) doesn't help?**

```
Singly:  To remove node B, need to find A first (traverse from head)
         (A) → (B) → (C)

Doubly:  Can remove B directly: B.prev.next = B.next
         (A) ⇄ (B) ⇄ (C)
```

Doubly linked list makes **removal O(1) once found**, but:

| Implementation | With Doubly LL | Improvement? |
|----------------|----------------|--------------|
| Unsorted | O(n) scan + O(1) remove | No - scan is still O(n) |
| Sorted | O(n) find position + O(1) insert | No - traverse is still O(n) |

The bottleneck is **finding** the element, not removing it. This is why **Heaps** are preferred - O(log n) for both operations.

**Usage:**
```python
from stack_queue.queue import PriorityQueueSortedList

pq = PriorityQueueSortedList()
pq.enqueue("Low", 3)
pq.enqueue("High", 1)
pq.enqueue("Urgent", 0)
print(pq.dequeue())  # Urgent (priority 0)
print(pq.dequeue())  # High (priority 1)
```

**Heap-based Priority Queue (recommended for balanced performance):**
```python
from heaps.binary_heap import PriorityQueueHeap

pq = PriorityQueueHeap()
pq.enqueue("Low", 3)
pq.enqueue("High", 1)
pq.enqueue("Urgent", 0)
print(pq.dequeue())  # Urgent (priority 0)
print(pq.dequeue())  # High (priority 1)
```

### 5. Trees

#### 5.1 Binary Tree

A binary tree is a hierarchical data structure where each node has **at most 2 children** (left and right).

```
       Root
       /  \
    Left   Right
    /  \
 Child  Child
```

**Key terms:**
- **Root**: Top node (no parent)
- **Leaf**: Node with no children
- **Height**: Longest path from root to leaf
- **Depth**: Distance from root to a node

#### Types of Binary Trees

| Type | Definition | Key Property |
|------|------------|--------------|
| Full | Every node has 0 or 2 children | No single-child nodes |
| Complete | All levels filled, last level fills left to right | Used for heaps |
| Perfect | All internal nodes have 2 children, leaves at same level | Full + Complete |
| Balanced | Height difference between subtrees ≤ 1 | O(log n) operations |
| Degenerate | Every node has only 1 child | O(n) operations (like linked list) |

```
Full:           Complete:       Perfect:        Degenerate:
    1               1               1               1
   / \             / \             / \               \
  2   3           2   3           2   3               2
 / \             / \ /           / \ / \               \
4   5           4  5 6          4  5 6  7               3
```

**Relationships:**
- Perfect = Full + Complete + all leaves at same level
- Balanced trees (AVL, Red-Black) guarantee O(log n) operations

#### 5.2 Binary Search Tree

A tree where left < root < right for all nodes.

| Operation | Description | Time (avg) | Time (worst) |
|-----------|-------------|------------|--------------|
| insert(data) | Add value | O(log n) | O(n) |
| search(data) | Find value | O(log n) | O(n) |
| delete(data) | Remove value | O(log n) | O(n) |
| find_min() | Get minimum | O(log n) | O(n) |
| find_max() | Get maximum | O(log n) | O(n) |

**Traversal Methods:**

| Method | Order | Use Case |
|--------|-------|----------|
| inorder() | Left, Root, Right | Sorted output (ascending) |
| reverse_inorder() | Right, Root, Left | Sorted output (descending) |
| preorder() | Root, Left, Right | Copy tree |
| postorder() | Left, Right, Root | Delete tree |
| level_order() | BFS by level | Level-by-level processing |

**Why these use cases?**

```
Example BST:
    4
   / \
  2   6
 / \ / \
1  3 5  7
```

| Traversal | Output | Reason |
|-----------|--------|--------|
| Inorder | 1, 2, 3, 4, 5, 6, 7 | Left first gives ascending order |
| Reverse Inorder | 7, 6, 5, 4, 3, 2, 1 | Right first gives descending order |
| Preorder | 4, 2, 1, 3, 6, 5, 7 | Root first, so parent created before children |
| Postorder | 1, 3, 2, 5, 7, 6, 4 | Children first, so they are deleted before parent |
| Level Order | 4, 2, 6, 1, 3, 5, 7 | Processes by depth, useful for shortest path |

**Inorder vs Reverse Inorder:**

Inorder is usually enough because for descending order, you can simply reverse the list:
```python
ascending = bst.inorder()        # [1, 2, 3, 4, 5]
descending = bst.inorder()[::-1] # [5, 4, 3, 2, 1]
```

| Scenario | Use |
|----------|-----|
| Need sorted ascending | inorder() |
| Need sorted descending | inorder()[::-1] (simpler) |
| Need top K largest (early termination) | reverse_inorder |
| Memory constrained (can't store list) | reverse_inorder |

**Why BFS for Level Order (not height + nodes_at_distance)?**

Level order can be implemented two ways:
```python
# BFS approach (used) - O(n) time
queue = [root]
while queue:
    node = queue.pop(0)
    visit(node)
    queue.append(node.left, node.right)

# Height approach - O(n²) time
for i in range(height + 1):
    nodes_at_distance(i)  # traverses from root each time
```

| Approach | Time | Why |
|----------|------|-----|
| BFS (queue) | O(n) | Each node visited once |
| height + nodes_at_distance | O(n²) | Traverses from root for EACH level |

BFS is more efficient because it visits each node exactly once, while the height approach re-traverses from root for every level.

**Recursion vs Iteration in Trees:**

| Path Type | Use | Space | Example |
|-----------|-----|-------|---------|
| Single direction (left OR right) | Iterative | O(1) | find_min, find_max |
| Both directions (left AND right) | Recursive | O(h) | traversals, height |

```
        50
       /  \
      30   70
     /
    20

find_min: 50 → 30 → 20 (single path, use iterative)
inorder:  visits ALL nodes (both branches, use recursive)
```

Why? Recursion uses the call stack to "remember" where to return after visiting a branch. For single-direction traversal, there's nothing to remember, so iterative is more efficient.

**nodes_at_distance() - Educational Purpose:**

The `nodes_at_distance(k)` method returns all nodes at distance k from the root. This method has **limited practical use** in BST but is included for educational purposes to understand tree depth.

```
        50          <- distance 0
       /  \
      30   70       <- distance 1
     / \   / \
    20 40 60 80     <- distance 2

nodes_at_distance(0)  → [50]
nodes_at_distance(1)  → [30, 70]
nodes_at_distance(2)  → [20, 40, 60, 80]
```

| Use Case | Description |
|----------|-------------|
| Get nodes at specific depth | Find all nodes exactly k levels from root |
| Find cousins | Nodes at same level but different parent |
| Tree visualization | Print specific level of tree |

| Need | Better Method |
|------|---------------|
| All levels | `level_order()` - O(n) |
| Single level only | `nodes_at_distance(k)` - O(n) |
| Nodes grouped by level | Modify `level_order()` to return nested list |

*Note: For most practical use cases, `level_order()` is preferred as it processes all levels in a single O(n) traversal.*

**Usage:**
```python
from trees.binary_tree import BinarySearchTree

bst = BinarySearchTree()
for val in [50, 30, 70, 20, 40]:
    bst.insert(val)

print(bst.inorder())      # [20, 30, 40, 50, 70]
print(bst.search(30))     # TreeNode with data=30
print(bst.height())       # 2
```

#### 5.3 AVL Tree (Self-Balancing BST)

An AVL tree is a self-balancing Binary Search Tree where the height difference between left and right subtrees is at most 1 for every node.

**Why AVL Tree?**

```
BST with ascending insertions (1,2,3,4,5,6,7):     AVL with same insertions:

1                                                        4
 \                                                      / \
  2                                                    2   6
   \                                                  / \ / \
    3                 Height: 6 (degenerate)         1  3 5  7    Height: 2
     \                Operations: O(n)
      4                                              Operations: O(log n)
       \
        5
         \
          6
           \
            7
```

| Operation | BST (worst) | AVL (guaranteed) |
|-----------|-------------|------------------|
| insert() | O(n) | O(log n) |
| delete() | O(n) | O(log n) |
| search() | O(log n) | O(log n) |

**Balance Factor:**

```
Balance Factor = height(left subtree) - height(right subtree)

| Balance Factor | Meaning |
|----------------|---------|
| -1, 0, 1 | Balanced |
| > 1 | Left-heavy (needs rotation) |
| < -1 | Right-heavy (needs rotation) |
```

**Rotations:**

AVL uses 4 types of rotations to maintain balance:

| Case | Condition | Rotation | Example |
|------|-----------|----------|---------|
| Left-Left (LL) | balance > 1 and left child is left-heavy | Right rotation | Insert 30, 20, 10 |
| Right-Right (RR) | balance < -1 and right child is right-heavy | Left rotation | Insert 10, 20, 30 |
| Left-Right (LR) | balance > 1 and left child is right-heavy | Left then Right | Insert 30, 10, 20 |
| Right-Left (RL) | balance < -1 and right child is left-heavy | Right then Left | Insert 10, 30, 20 |

**Rotation Diagrams:**

```
Right Rotation (LL case):        Left Rotation (RR case):

    y                x               x                  y
   / \              / \             / \                / \
  x   C    -->     A   y           A   y     -->      x   C
 / \                  / \             / \            / \
A   B                B   C           B   C          A   B
```

```
Left-Right (LR case):                Right-Left (RL case):

    z               z               x           z               z               x
   /               /               / \           \               \             / \
  y      -->      x      -->      y   z           y     -->       x    -->    z   y
   \             /                                /                 \
    x           y                                x                   y
```

**Why AVL Insert Returns Node but BST Doesn't:**

```python
# BST: Direct assignment, no return needed
def _insert_recursive(self, node, data):
    if data < node.data:
        if node.left is None:
            node.left = TreeNode(data)  # Just link new node
        else:
            self._insert_recursive(node.left, data)
    # node is still the root - no return needed

# AVL: Must return because rotation can change subtree root
def _insert_recursive(self, node, data):
    if node is None:
        return AVLNode(data)
    if data < node.data:
        node.left = self._insert_recursive(node.left, data)
    else:
        node.right = self._insert_recursive(node.right, data)
    return self._rebalance(node)  # May return DIFFERENT node!
```

```
Insert 3 into AVL:       After rotation, root changes!

    5                        5                      4
   /          insert 3      /        rotate        / \
  4           -------->    4         ------>      3   5
                          /
                         3

Subtree root changed from 5 to 4!
Parent must update its child pointer to new root.
```

| Tree | After Insert | Root Changes? | Need Return? |
|------|--------------|---------------|--------------|
| BST | Just links new node | No | No |
| AVL | May rotate subtree | Yes | Yes |

**Summary:** BST structure only grows (no reshape), AVL rotations can change subtree root - parent needs the new root to update its pointer.

**AVL vs BST Method Comparison:**

| Method | Same Logic? | Difference |
|--------|-------------|------------|
| `insert()` | No | AVL: returns node + rebalances |
| `delete()` | No | AVL: returns node + rebalances |
| `search()` | Yes | Identical - just traverse |
| `find_min()` | Yes | Identical - go left |
| `find_max()` | Yes | Identical - go right |
| `inorder()` | Yes | Identical traversal |
| `preorder()` | Yes | Identical traversal |
| `postorder()` | Yes | Identical traversal |
| `level_order()` | Yes | Identical BFS |
| `height()` | Slight | AVL: O(1) stored in node, BST: O(n) computed |

**AVL-Only Methods (not in BST):**

| Method | Purpose | Time |
|--------|---------|------|
| `_get_balance_factor()` | Calculate balance = height(left) - height(right) | O(1) |
| `_update_height()` | Update node's height after changes | O(1) |
| `_rotate_left()` | Left rotation for RR/RL cases | O(1) |
| `_rotate_right()` | Right rotation for LL/LR cases | O(1) |
| `_rebalance()` | Check balance and apply rotations if needed | O(1) |
| `is_balanced()` | Verify tree is balanced (should always be True) | O(n) |

**How `_rebalance()` Works:**

```
Step 1: Update height of CURRENT node only
Step 2: Calculate balance factor of CURRENT node (left_height - right_height)
Step 3: Check if CURRENT node is unbalanced (|balance| > 1)
Step 4: If unbalanced, determine case and rotate:

Balance > 1 (Left-heavy):
├── Left child balance >= 0  →  LL Case  →  Right rotation
└── Left child balance < 0   →  LR Case  →  Left(left) then Right(node)

Balance < -1 (Right-heavy):
├── Right child balance <= 0 →  RR Case  →  Left rotation
└── Right child balance > 0  →  RL Case  →  Right(right) then Left(node)
```

**Height Update Process (Bottom-Up):**

`_rebalance()` is called on each node as recursion unwinds, so all affected ancestors get updated one at a time:

```
Delete 20 from tree:

        50              Step 1: Go DOWN to find 20
       /  \
      30   70           Step 2: Delete 20, return None
     /
    20 <-- delete

Coming back UP (recursion unwinds):

        50              3. _rebalance(50) → _update_height(50)
       /  \                height = max(0, 0) + 1 = 1
      30   70           2. _rebalance(30) → _update_height(30)
     /                     height = max(-1, -1) + 1 = 0
   None                 1. Return None (20 deleted)

Order of calls:
delete(50, 20)
  └─> delete(30, 20)
        └─> delete(20, 20) → returns None
        └─> _rebalance(30) → updates height of 30 ONLY
  └─> _rebalance(50) → updates height of 50 ONLY
```

**Why bottom-up works:**
- Children are processed before parents
- When updating parent's height, children's heights are already correct
- `height = max(left.height, right.height) + 1` gives accurate result

```python
def _rebalance(self, node):
    # Step 1: Update height
    self._update_height(node)

    # Step 2: Calculate balance factor
    balance = self._get_balance_factor(node)

    # Step 3 & 4: Check and apply rotations

    # Left-heavy (balance > 1)
    if balance > 1:
        # LR Case: Left child is right-heavy
        if self._get_balance_factor(node.left) < 0:
            node.left = self._rotate_left(node.left)  # First rotation
        # LL Case (or after LR adjustment)
        return self._rotate_right(node)

    # Right-heavy (balance < -1)
    if balance < -1:
        # RL Case: Right child is left-heavy
        if self._get_balance_factor(node.right) > 0:
            node.right = self._rotate_right(node.right)  # First rotation
        # RR Case (or after RL adjustment)
        return self._rotate_left(node)

    # Already balanced
    return node
```

**Rotation Decision Table:**

| Balance | Child Balance | Case | Rotations |
|---------|---------------|------|-----------|
| > 1 | left >= 0 | LL | Right(node) |
| > 1 | left < 0 | LR | Left(left) → Right(node) |
| < -1 | right <= 0 | RR | Left(node) |
| < -1 | right > 0 | RL | Right(right) → Left(node) |

**Key Differences:**

1. **Node structure:**
```python
# BST Node: 2 pointers
class TreeNode:
    data, left, right

# AVL Node: 2 pointers + height
class AVLNode:
    data, left, right, height  # Extra field
```

2. **Height retrieval:**
```python
# BST: O(n) - must traverse entire tree
def height(self):
    return self._height_recursive(self.root)

# AVL: O(1) - stored in node, updated during insert/delete
def height(self):
    return self._get_height(self.root)
```

**Why AVL Stores Height but BST Doesn't:**

AVL needs height to calculate balance factor after every insert/delete:
```python
balance = height(left) - height(right)
# If |balance| > 1, need rotation
```

| Approach | Balance Check | When Used |
|----------|---------------|-----------|
| Compute height each time | O(n) | Too slow for every insert/delete |
| Store height in node | O(1) | Efficient - just read stored value |

```python
# Without stored height - O(n) each check
def _get_balance_factor(self, node):
    left_height = self._compute_height(node.left)   # O(n) traversal
    right_height = self._compute_height(node.right) # O(n) traversal
    return left_height - right_height

# With stored height - O(1) each check
def _get_balance_factor(self, node):
    return self._get_height(node.left) - self._get_height(node.right)  # O(1)
```

**Why BST doesn't store height:**
- BST doesn't check balance - no rebalancing operations
- Height is rarely needed (only if user explicitly calls `height()`)
- Computing O(n) once when needed is acceptable
- Saves memory by not storing extra field in every node

**AVL height is updated:**
1. After inserting a node
2. After deleting a node
3. After each rotation

**Summary:** Only `insert` and `delete` differ (rebalancing logic). Search and traversals are identical since they don't modify tree structure.

**When to Use AVL vs BST:**

| Scenario | Use |
|----------|-----|
| Frequent insertions/deletions, rare searches | BST (less rotation overhead) |
| Frequent searches, data changes rarely | AVL (guaranteed O(log n) search) |
| Data inserted in sorted order | AVL (BST degenerates) |
| Need guaranteed performance | AVL |

**Usage:**
```python
from trees.avl_tree import AVLTree

avl = AVLTree()

# Insert in ascending order (would be degenerate in BST)
for val in [1, 2, 3, 4, 5, 6, 7]:
    avl.insert(val)

print(avl.height())       # 2 (BST would be 6)
print(avl.inorder())      # [1, 2, 3, 4, 5, 6, 7]
print(avl.is_balanced())  # True

avl.delete(4)
print(avl.is_balanced())  # True (auto-rebalanced)
```

### 6. Heaps

A binary heap is a complete binary tree that satisfies the heap property.

| Type | Property | Root |
|------|----------|------|
| Min-Heap | parent <= children | Smallest element |
| Max-Heap | parent >= children | Largest element |

**Time Complexity:**

| Operation | Time | Space | Description |
|-----------|------|-------|-------------|
| `insert()` | O(log n) | O(1) | Add element, heapify up |
| `extract_min/max()` | O(log n) | O(1) | Remove root, heapify down |
| `peek()` | O(1) | O(1) | View root element |
| `heapify()` | O(n) | O(n) | Build heap (creates copy) |
| `heapify_inplace()` | O(n) | O(1) | Build heap (modifies original) |
| `heapify_naive()` | O(n log n) | O(n) | Build heap by inserting one by one |
| `heap_sort()` | O(n log n) | O(n) | Sort array using heap |
| `is_empty()` | O(1) | O(1) | Check if empty |

**Why Heap Sort is O(n log n):**

| Step | Operation | Time |
|------|-----------|------|
| 1. Build heap | `heapify()` | O(n) |
| 2. Extract n elements | n × `extract_min/max()` | n × O(log n) = **O(n log n)** |

The key is that **each extraction is O(log n)**, not O(1):

```
extract_min() / extract_max():
1. Remove root             - O(1)
2. Move last to root       - O(1)
3. heapify_down()          - O(log n)  ← expensive part
```

`heapify_down()` must potentially traverse from root to leaf to restore heap property. Since heap height = log n, this is O(log n).

**Total:** O(n) + O(n log n) = **O(n log n)**

The O(n) from building is dominated by O(n log n) from extractions.

```
Example (n=7 elements):
Extract 1: heapify_down traverses up to 3 levels
Extract 2: heapify_down traverses up to 3 levels
Extract 3: heapify_down traverses up to 2 levels
...
Total: ~n × log n operations
```

**MinHeap vs MaxHeap heap_sort:**

| Heap Type | Output Order |
|-----------|--------------|
| MinHeap | Ascending (smallest first) |
| MaxHeap | Descending (largest first) |

**Why Use Python List for Binary Heap:**

A binary heap is always a **complete binary tree**, which maps perfectly to a list:

```
Tree view:              List view:

        1               [1, 3, 2, 7, 4, 5, 6]
       / \               0  1  2  3  4  5  6  (indices)
      3   2
     / \ / \
    7  4 5  6
```

**Index Formulas:**
```python
Parent of index i:      (i - 1) // 2
Left child of index i:  2 * i + 1
Right child of index i: 2 * i + 2
```

**Example:**
```
Index 0 (value 1):
  - Left child: 2*0+1 = 1 (value 3)
  - Right child: 2*0+2 = 2 (value 2)

Index 1 (value 3):
  - Parent: (1-1)//2 = 0 (value 1)
  - Left child: 2*1+1 = 3 (value 7)
```

**Why List is Ideal (not Tree Nodes):**

| Implementation | Pros | Cons |
|----------------|------|------|
| List/Array | O(1) parent/child access, cache-friendly, less memory | Only works for complete trees |
| Tree Nodes | Flexible structure | Extra pointers, more memory, slower |

Since binary heap is always complete, list is the preferred implementation.

**Why No `size` Attribute:**

Unlike linked list implementations, the heap class doesn't need a `size` attribute because Python lists internally store their length:

```
Python list internal structure:
┌─────────────────────────┐
│  ob_size: 5  ← stored!  │  len() just reads this - O(1)
│  allocated: 8           │
│  *ob_item → [1,2,3,4,5] │
└─────────────────────────┘
```

| Data Structure | Has `size` attr? | Why? |
|----------------|------------------|------|
| Linked List | Yes | No internal length, traversal would be O(n) |
| Stack (linked) | Yes | No internal length, traversal would be O(n) |
| Heap (list-based) | No | `len(list)` is O(1), reads internal `ob_size` |

Using `len(self.heap)` is O(1), so storing a separate `size` would be redundant.

**Why `extract_min/max()` not `delete()`:**

`extract_min()` / `extract_max()` is the conventional naming for heap's delete operation:

| Name | Implies | Used in |
|------|---------|---------|
| `extract_min/max()` | Remove root AND return value | Textbooks (CLRS), algorithms |
| `pop()` | Remove and return | Python `heapq` |
| `delete()` | Just remove, unclear what | Generic structures |

Why `extract` is better:
- **Clear which element:** `extract_min()` vs `delete()` (delete what?)
- **Implies value returned:** "Extract" = take out and return
- **Distinguishes from specific value deletion:** `extract_min()` always removes root, `delete(value)` would need O(n) search
- **Matches standard CS terminology:** CLRS textbook uses `EXTRACT-MIN`, `EXTRACT-MAX`

```python
# Clear and explicit
value = min_heap.extract_min()  # Removes and returns minimum
value = max_heap.extract_max()  # Removes and returns maximum

# Confusing
value = heap.delete()  # Delete what? Return what?
```

**Heapify Up vs Heapify Down:**

| Operation | When Used | Direction |
|-----------|-----------|-----------|
| `_heapify_up()` | After insert | Bubble up from leaf to root |
| `_heapify_down()` | After extract | Sink down from root to leaf |

```
Insert 0 into MinHeap [1, 3, 2]:     Extract min from MinHeap [1, 3, 2]:

Step 1: Append          Step 1: Remove root, move last to root
[1, 3, 2, 0]            [2, 3]
        1                       2
       / \                     /
      3   2                   3
     /
    0  <-- new              Step 2: Heapify down
                            [2, 3] → already valid
Step 2: Heapify up
[1, 3, 2, 0]
     ↓
[1, 0, 2, 3]
     ↓
[0, 1, 2, 3]  <-- 0 bubbled up to root
```

**heapify Methods Comparison:**

| Method | Time | Space | Modifies Original |
|--------|------|-------|-------------------|
| `heapify()` | O(n) | O(n) | No (creates copy) |
| `heapify_inplace()` | O(n) | O(1) | Yes |
| `heapify_naive()` | O(n log n) | O(n) | No (creates new) |

```python
# heapify() - O(n) time, O(n) space (creates copy)
def heapify(self, array):
    self.heap = array.copy()  # Copy - original unchanged
    for i in range(len(self.heap) // 2 - 1, -1, -1):
        self._heapify_down(i)

# heapify_inplace() - O(n) time, O(1) space (modifies original)
def heapify_inplace(self, array):
    self.heap = array  # Reference - original modified
    for i in range(len(self.heap) // 2 - 1, -1, -1):
        self._heapify_down(i)

# heapify_naive() - O(n log n) time, O(n) space
def heapify_naive(self, array):
    self.heap = []
    for item in array:        # n items
        self.insert(item)     # O(log n) each
```

**In-place vs Copy behavior:**
```python
# heapify() - original unchanged
arr = [5, 3, 8, 1, 2]
heap.heapify(arr)
print(arr)        # [5, 3, 8, 1, 2]  <-- unchanged
print(heap.heap)  # [1, 2, 8, 3, 5]  <-- heap's copy

# heapify_inplace() - original modified
arr = [5, 3, 8, 1, 2]
heap.heapify_inplace(arr)
print(arr)        # [1, 2, 8, 3, 5]  <-- modified!
print(heap.heap)  # [1, 2, 8, 3, 5]  <-- same object
print(arr is heap.heap)  # True
```

**Why heapify() is O(n) not O(n log n):**

`heapify()` starts from last non-leaf and works up:
- Half the nodes are leaves (no work needed)
- Quarter of nodes do 1 swap max
- Eighth of nodes do 2 swaps max
- Sum = O(n)

```
        1         Level 0: 1 node,  up to 3 swaps (but only 1 node)
       / \
      3   2       Level 1: 2 nodes, up to 2 swaps each
     / \ / \
    7  4 5  6     Level 2: 4 nodes, up to 1 swap each
                  Level 3: (leaves) 0 swaps

Most nodes are near the bottom where they do FEWER swaps.
```

**Mathematical proof:** Sum = n/4 × 1 + n/8 × 2 + n/16 × 3 + ... = O(n)

**Usage:**
```python
from heaps.binary_heap import MinHeap, MaxHeap

# MinHeap - smallest at root
min_heap = MinHeap()
min_heap.insert(5)
min_heap.insert(3)
min_heap.insert(8)
print(min_heap.peek())        # 3 (smallest)
print(min_heap.extract_min()) # 3

# MaxHeap - largest at root
max_heap = MaxHeap()
max_heap.insert(5)
max_heap.insert(3)
max_heap.insert(8)
print(max_heap.peek())        # 8 (largest)
print(max_heap.extract_max()) # 8

# Build heap from array - O(n)
arr = [5, 3, 8, 1, 2, 9, 4]
min_heap.heapify(arr)
print(min_heap.heap)  # [1, 2, 4, 3, 5, 9, 8]
```

**Heap vs Priority Queue:**

The heap implementations can be used as efficient priority queues:

| Implementation | enqueue | dequeue | peek |
|----------------|---------|---------|------|
| Unsorted List | O(1) | O(n) | O(n) |
| Sorted List | O(n) | O(1) | O(1) |
| Binary Heap | O(log n) | O(log n) | O(1) |

Heap provides balanced O(log n) for both insert and remove operations.

#### 6.1 PriorityQueueHeap

A priority queue implementation using MinHeap (lower value = higher priority).

| Method | Time | Description |
|--------|------|-------------|
| `enqueue(data, priority)` | O(log n) | Add element with priority |
| `dequeue()` | O(log n) | Remove highest priority element |
| `peek()` | O(1) | View highest priority element |
| `is_empty()` | O(1) | Check if empty |

**How it works:**

```
enqueue("Low", 3), enqueue("High", 1), enqueue("Urgent", 0):

Internal heap (sorted by priority):
        (0, Urgent)      ← root is always highest priority
       /           \
(1, High)    (3, Low)

dequeue() returns: "Urgent" (priority 0)
dequeue() returns: "High" (priority 1)
dequeue() returns: "Low" (priority 3)
```

**Usage:**
```python
from heaps.binary_heap import PriorityQueueHeap

pq = PriorityQueueHeap()
pq.enqueue("Low", 3)
pq.enqueue("High", 1)
pq.enqueue("Urgent", 0)

print(pq.peek())     # Urgent
print(pq.dequeue())  # Urgent (priority 0)
print(pq.dequeue())  # High (priority 1)
print(pq.dequeue())  # Low (priority 3)
```

**When to use which priority queue:**

| Scenario | Best Implementation |
|----------|---------------------|
| Many inserts, few removals | Unsorted List - O(1) insert |
| Few inserts, many removals | Sorted List - O(1) remove |
| Balanced insert/remove | **Heap** - O(log n) both |
| Unknown workload | **Heap** - consistent performance |

### 7. Tries

A Trie (prefix tree) is a tree-like data structure used for efficient retrieval of keys in a dataset of strings. Common use cases include **autocomplete**, **spell checking**, and **IP routing**.

```
Trie storing "app", "apple", "application", "banana":

        root
       /    \
      a      b
      |      |
      p      a
      |      |
      p*     n
     / \     |
    l   i    a
    |   |    |
    e*  c    n
        |    |
        a    a*
        |
        t
        |
        i
        |
        o
        |
        n*

* = end of word (is_end_of_word = True)
```

**Four implementations with different trade-offs:**

| Class | Node Storage | Child Lookup | Memory | Why Memory Pattern |
|-------|--------------|--------------|--------|-------------------|
| `Trie` | `dict[str, Node]` | O(1) hash | Dynamic | Dict grows with children, has hash table overhead |
| `TrieLCRS` | child Node + sibling Node | O(k) scan | Minimal | Just pointers, no container overhead |
| `TrieArray` | `list[Node] * 26` | O(1) index | Fixed 26/node | Pre-allocates 26 slots, wastes space if sparse |
| `TrieList` | `list[Node]` dynamic | O(k) scan | Dynamic | List grows with children, less overhead than dict |

*k = number of children at each node*

#### 7.1 Trie Operations

**Time Complexity (where m = word length, k = avg children per node):**

| Operation | Trie (Dict) | TrieArray | TrieLCRS | TrieList |
|-----------|-------------|-----------|----------|----------|
| `insert(word)` | O(m) | O(m) | O(m × k) | O(m × k) |
| `search(word)` | O(m) | O(m) | O(m × k) | O(m × k) |
| `delete(word)` | O(m) | O(m) | O(m × k) | O(m × k) |
| `starts_with(prefix)` | O(m) | O(m) | O(m × k) | O(m × k) |
| `auto_complete(prefix)` | O(m + n) | O(m + n) | O(m × k + n) | O(m × k + n) |
| `get_all_words()` | O(n × m) | O(n × m) | O(n × m) | O(n × m) |

*n = number of words in result*

**insert(word) - How it works:**

```
Insert "app" into empty trie:

Step 1: Start at root        Step 2: Add 'a'         Step 3: Add 'p'         Step 4: Add 'p', mark end

  root                         root                    root                    root
                               |                       |                       |
                               a                       a                       a
                                                       |                       |
                                                       p                       p
                                                                               |
                                                                               p*

* = is_end_of_word = True
```

```python
def insert(self, word: str) -> None:
    node = self.root
    for char in word:
        # Find or create child for this character
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    # Mark the last node as end of word
    if not node.is_end_of_word:
        node.is_end_of_word = True
        self.word_count += 1
```

**How child lookup differs by implementation:**

```python
# Dict (Trie) - O(1) hash lookup
if char not in node.children:          # Hash lookup
    node.children[char] = TrieNode()   # Direct assignment

# Array (TrieArray) - O(1) index lookup
index = ord(char) - ord('a')           # Convert 'a'->0, 'b'->1, etc.
if node.children[index] is None:       # Direct index access
    node.children[index] = TrieNodeArray()

# LCRS/List - O(k) linear scan
child = None
for c in node.children:                # Must scan all children
    if c.char == char:
        child = c
        break
if child is None:
    child = TrieNodeList(char)
    node.children.append(child)
```

**delete(word) - How it works:**

Delete is more complex because we need to:
1. Find the word and unmark `is_end_of_word`
2. Clean up unused nodes (nodes that are not end of any word and have no children)

```
Delete "app" from trie containing "app", "apple":

Before:                 After:
  root                    root
  |                       |
  a                       a
  |                       |
  p                       p
  |                       |
  p* ← unmark             p  ← no longer marked, but keep (path to 'l')
  |                       |
  l                       l
  |                       |
  e*                      e*

The 'p' node is kept because it's still part of "apple".
```

```
Delete "apple" from trie containing only "apple":

Before:                 After cleanup:
  root                    root (empty)
  |
  a    ← delete (no children, not end of word)
  |
  p    ← delete (no children, not end of word)
  |
  p    ← delete (no children, not end of word)
  |
  l    ← delete (no children, not end of word)
  |
  e*   ← unmark first, then delete (no children)

All nodes removed because none are needed.
```

```python
def _delete_recursive(self, node: TrieNode, word: str, index: int) -> bool:
    # Base case: reached end of word
    if index == len(word):
        if not node.is_end_of_word:
            return False  # Word doesn't exist
        node.is_end_of_word = False
        self.word_count -= 1
        # Return True if node can be deleted (no children)
        return not self._has_children(node)

    char = word[index]
    if char not in node.children:
        return False  # Word doesn't exist

    # Recurse to next character
    should_delete_child = self._delete_recursive(
        node.children[char], word, index + 1
    )

    # If child should be deleted, remove it
    if should_delete_child:
        del node.children[char]
        # Current node can be deleted if no children and not end of word
        return not self._has_children(node) and not node.is_end_of_word

    return False
```

**auto_complete(prefix) - Autocomplete:**

This is the key feature of tries - finding all words that start with a given prefix.

```
Trie with "app", "apple", "application", "apply", "banana":

auto_complete("app") returns: ["app", "apple", "application", "apply"]

Step 1: Navigate to prefix "app"
        root
        |
        a
        |
        p
        |
        p* ← start collecting from here

Step 2: Collect all words from this node (DFS)
        p* ← "app" (is_end_of_word)
       /|\
      l i y
      |  |  |
      e* c  * ← "apply"
         |
         a
         |
         t
         |
         i
         |
         o
         |
         n* ← "application"

Result: ["app", "apple", "application", "apply"]
```

```python
def auto_complete(self, prefix: str) -> list[str]:
    # Step 1: Find the node at end of prefix
    node = self._find_node(prefix)
    if node is None:
        return []  # Prefix doesn't exist

    # Step 2: Collect all words from this node
    words: list[str] = []
    self._collect_words(node, prefix, words)
    return words

def _collect_words(self, node: TrieNode, prefix: str, words: list[str]) -> None:
    # If this node marks end of a word, add it
    if node.is_end_of_word:
        words.append(prefix)
    # Recursively collect from all children
    for char, child in node.children.items():
        self._collect_words(child, prefix + char, words)
```

**Why Trie is efficient for autocomplete:**

| Approach | Time to find words with prefix "app" |
|----------|--------------------------------------|
| Linear search through word list | O(n × m) - check every word |
| Binary search (sorted list) | O(log n + k × m) - find range, then collect |
| Trie | O(p + k × m) - navigate prefix, then collect |

*n = total words, m = avg word length, p = prefix length, k = matching words*

For autocomplete with many words but few matches, Trie is significantly faster because it only visits relevant nodes.

#### 7.2 Implementation Comparison

**Node Structure Comparison:**

```python
# Dict-based (Trie) - character stored as dict key
class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False

# Array-based (TrieArray) - character encoded as index (a=0, b=1, ...)
class TrieNodeArray:
    def __init__(self):
        self.children: list[Optional[TrieNodeArray]] = [None] * 26
        self.is_end_of_word: bool = False

# LCRS (TrieLCRS) - character stored in node, siblings linked
class TrieNodeLCRS:
    def __init__(self, char: str = ""):
        self.char: str = char
        self.child: Optional[TrieNodeLCRS] = None
        self.sibling: Optional[TrieNodeLCRS] = None
        self.is_end_of_word: bool = False

# List-based (TrieList) - character stored in node, children in list
class TrieNodeList:
    def __init__(self, char: str = ""):
        self.char: str = char
        self.children: list[TrieNodeList] = []
        self.is_end_of_word: bool = False
```

**Why different structures store character differently:**

| Structure | Where char stored | Why |
|-----------|-------------------|-----|
| Dict | Key in parent's dict | `children['a']` gives direct O(1) access |
| Array | Index in parent's array | `children[0]` = 'a', O(1) access |
| LCRS | In node itself | Must scan siblings to find, needs char to compare |
| List | In node itself | Must scan list to find, needs char to compare |

**Space Complexity:**

| Implementation | Memory per Node | Total Space | Notes |
|----------------|-----------------|-------------|-------|
| Dict | ~56 bytes + dict overhead | O(total unique chars) | Dict has overhead but only stores existing children |
| Array | ~232 bytes (26 pointers) | O(nodes × 26) | Fixed 26 slots even if empty |
| LCRS | ~40 bytes (3 pointers + char) | O(total unique chars) | Most memory efficient |
| List | ~56 bytes + list overhead | O(total unique chars) | Similar to Dict |

**When to use which:**

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| General purpose | `Trie` (Dict) | O(1) lookup, flexible character set |
| Fixed alphabet (a-z only) | `TrieArray` | Fastest lookup, predictable memory |
| Memory constrained | `TrieLCRS` | Minimal memory per node |
| Large sparse trie | `TrieLCRS` / `TrieList` | Array wastes 26 slots per node |
| Millions of words | `TrieLCRS` | Memory savings add up significantly |
| Learning/educational | `TrieLCRS` | Understand tree pointer structures |
| Simple code, any charset | `TrieList` | Simpler than LCRS, similar performance |

**Performance vs Memory Trade-off:**

```
                    Lookup Speed
                         ▲
                         │
          TrieArray ●    │    ● Trie (Dict)
             (O(1))      │      (O(1))
                         │
                         │
                         │
                         │
          TrieList ●─────┼────● TrieLCRS
            (O(k))       │      (O(k))
                         │
                         └──────────────────► Memory Efficiency

Dict/Array: Fast but more memory
LCRS/List: Slower but less memory
```

**Usage:**

```python
from tries.trie import Trie, TrieLCRS, TrieArray, TrieList

# Dictionary-based (recommended for most cases)
trie = Trie()
trie.insert("apple")
trie.insert("app")
trie.insert("application")

print(trie.search("app"))           # True
print(trie.search("appl"))          # False (not a complete word)
print(trie.starts_with("app"))      # True
print(trie.auto_complete("app"))  # ['apple', 'app', 'application']

trie.delete("app")
print(trie.search("app"))           # False
print(trie.search("apple"))         # True (still exists)

# Array-based (for a-z only, fastest lookup)
trie_array = TrieArray()
trie_array.insert("hello")
trie_array.insert("help")
print(trie_array.auto_complete("hel"))  # ['hello', 'help']

# LCRS (minimal memory)
trie_lcrs = TrieLCRS()
trie_lcrs.insert("cat")
trie_lcrs.insert("car")
print(trie_lcrs.auto_complete("ca"))  # ['cat', 'car']
```

### 8. Graphs

A Graph is a non-linear data structure consisting of **vertices** (nodes) and **edges** (connections). Graphs are used to represent networks, relationships, and connections between entities.

```
Undirected Graph:              Directed Graph (Digraph):

    A --- B                        A --→ B
    |     |                        ↑     |
    |     |                        |     ↓
    C --- D --- E                  C ←-- D --→ E
```

**Graph Terminology:**

| Term | Description |
|------|-------------|
| Vertex (Node) | A point in the graph |
| Edge | Connection between two vertices |
| Directed | Edges have direction (A → B ≠ B → A) |
| Undirected | Edges are bidirectional (A — B = B — A) |
| Weighted | Edges have associated values (costs/distances) |
| Degree | Number of edges connected to a vertex |
| Path | Sequence of vertices connected by edges |
| Cycle | Path that starts and ends at the same vertex |
| Connected | Path exists between every pair of vertices |

**Two implementations with different trade-offs:**

| Class | Storage | Space | Edge Check | Get Neighbors | Best For |
|-------|---------|-------|------------|---------------|----------|
| `GraphAdjMatrix` | V×V matrix | O(V²) | O(1) | O(V) | Dense graphs |
| `GraphAdjList` | Dict of lists | O(V + E) | O(degree) | O(degree) | Sparse graphs |

#### 8.1 Graph Operations

**Time Complexity (V = vertices, E = edges):**

| Operation | Adjacency Matrix | Adjacency List |
|-----------|------------------|----------------|
| `add_vertex()` | O(V) | O(1) |
| `remove_vertex()` | O(V²) | O(V + E) |
| `add_edge()` | O(1) | O(1) |
| `remove_edge()` | O(1) | O(degree) |
| `has_edge()` | O(1) | O(degree) |
| `get_neighbors()` | O(V) | O(degree) |
| `bfs()` / `dfs()` | O(V²) | O(V + E) |
| `topological_sort()` | O(V²) | O(V + E) |
| `has_cycle()` | O(V²) | O(V + E) |
| `shortest_path()` | O(V² log V) | O((V + E) log V) |
| `shortest_distance()` | O(V² log V) | O((V + E) log V) |
| `minimum_spanning_tree()` | O(V² log V) | O((V + E) log V) |

*degree = number of edges connected to a vertex*

```
Example:
    A --- B --- C          Degrees:
    |                      - A: degree = 2 (edges to B, D)
    D                      - B: degree = 2 (edges to A, C)
                           - C: degree = 1 (edge to B)
                           - D: degree = 1 (edge to A)

O(degree) means scanning through a vertex's neighbor list.
For sparse graphs: degree ≈ small → O(degree) ≈ O(1)
For dense graphs: degree ≈ V → O(degree) ≈ O(V)
```

**Adjacency List in Dense Graphs (E ≈ V²):**

| Operation | Sparse (E << V²) | Dense (E ≈ V²) |
|-----------|------------------|----------------|
| `remove_vertex()` | O(V + E) | O(V²) |
| `remove_edge()` | O(degree) | O(V) |
| `has_edge()` | O(degree) | O(V) |
| `bfs()` / `dfs()` | O(V + E) | O(V²) |
| `topological_sort()` | O(V + E) | O(V²) |
| `has_cycle()` | O(V + E) | O(V²) |
| `shortest_path()` | O((V + E) log V) | O(V² log V) |
| `minimum_spanning_tree()` | O((V + E) log V) | O(V² log V) |

*In dense graphs, adjacency list loses its advantage over adjacency matrix.*

**Adjacency Matrix Representation:**

```
Graph:                    Adjacency Matrix:
    A --- B                     A   B   C   D
    |     |               A  [  0   1   1   0  ]
    |     |               B  [  1   0   0   1  ]
    C --- D               C  [  1   0   0   1  ]
                          D  [  0   1   1   0  ]

matrix[i][j] = 1 means edge exists from vertex i to vertex j
matrix[i][j] = 0 means no edge
For weighted graphs, store the weight instead of 1
```

**Adjacency List Representation:**

```
Graph:                    Adjacency List:
    A --- B               A: [B, C]
    |     |               B: [A, D]
    |     |               C: [A, D]
    C --- D               D: [B, C]

Each vertex stores a list of its neighbors
For weighted graphs, store (neighbor, weight) tuples
```

**BFS (Breadth-First Search) - Level by level traversal:**

```
        A                 BFS from A: A → B → C → D → E
       / \
      B   C               Uses: Queue (FIFO)
      |   |               Process: Visit all neighbors at current level
      D   |                         before moving to next level
       \ /
        E

Order: A, B, C, D, E (level by level)
```

```python
def bfs(self, start: str) -> list[str]:
    visited = set()
    result = []
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()      # FIFO - process in order added
        result.append(vertex)

        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result
```

**DFS (Depth-First Search) - Go deep before backtracking:**

```
        A                 DFS from A: A → B → D → E → C
       / \
      B   C               Uses: Stack (LIFO) or Recursion
      |   |               Process: Go as deep as possible,
      D   |                         then backtrack
       \ /
        E

Order: A, B, D, E, C (deep first)
```

```python
def dfs(self, start: str) -> list[str]:
    visited = set()
    result = []
    stack = [start]

    while stack:
        vertex = stack.pop()          # LIFO - process most recent
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)

            for neighbor in reversed(self.get_neighbors(vertex)):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result
```

**BFS vs DFS Comparison:**

| Aspect | BFS | DFS |
|--------|-----|-----|
| Data structure | Queue | Stack/Recursion |
| Order | Level by level | Deep then backtrack |
| Shortest path | Yes (unweighted) | No |
| Memory | O(V) worst case | O(V) worst case |
| Use case | Shortest path, level order | Cycle detection, topological sort |

**When to Mark Visited - BFS vs DFS:**

```python
# BFS: Mark BEFORE adding to queue (prevents duplicates)
if neighbor not in visited:
    visited.add(neighbor)      # Mark before enqueue
    queue.append(neighbor)

# DFS: Mark AFTER popping from stack (allows duplicates)
vertex = stack.pop()
if vertex not in visited:      # Check after pop
    visited.add(vertex)        # Mark after pop
```

| Strategy | When to mark | Duplicates | Used by |
|----------|--------------|------------|---------|
| Mark before adding | When enqueueing/pushing | No | BFS (typically) |
| Mark after removing | When dequeuing/popping | Possible | DFS (typically) |

*Why the difference?* There's no fundamental requirement - both strategies work for both algorithms. DFS typically uses "mark after pop" because it mirrors recursive DFS, where we mark visited when entering a function (starting to process), not when adding to the call stack.

*Both are correct.* "Mark before" is more memory efficient (no duplicates), but "mark after" follows the recursive pattern more naturally.

**DFS "mark after pop" (current implementation):**

```python
stack.append(start)
while stack:
    vertex = stack.pop()
    if vertex not in visited:      # Check after pop
        visited.add(vertex)        # Mark after pop
        result.append(vertex)
        for neighbor in self.get_neighbors(vertex):
            if neighbor not in visited:
                stack.append(neighbor)  # May add duplicates
```

**DFS "mark before push" (more memory efficient):**

```python
visited.add(start)                 # Mark start before push
stack.append(start)
while stack:
    vertex = stack.pop()
    result.append(vertex)          # No check needed - already visited
    for neighbor in self.get_neighbors(vertex):
        if neighbor not in visited:
            visited.add(neighbor)  # Mark before push - no duplicates
            stack.append(neighbor)
```

The "mark before push" approach prevents the same vertex from being pushed multiple times, reducing memory usage.

#### 8.2 Topological Sort

Topological sorting is a linear ordering of vertices in a **Directed Acyclic Graph (DAG)** such that for every directed edge (u → v), vertex u comes **before** vertex v in the ordering.

```
Course Prerequisites (DAG):

  Math → Physics → Quantum
    ↓
  Stats → ML

Topological Order: Math → Stats → Physics → ML → Quantum
(or: Math → Physics → Stats → ML → Quantum - multiple valid orders exist)

Rule: Prerequisites must come before dependent courses.
```

**Requirements:**
- Must be a **directed** graph
- Must be **acyclic** (no cycles) - otherwise impossible

```
Valid (DAG):              Invalid (has cycle):
  A → B → C                 A → B
      ↓                     ↑   ↓
      D                     D ← C

Topological order exists   No valid ordering - cycle detected
```

**Common Use Cases:**

| Use Case | Vertices | Edges |
|----------|----------|-------|
| Course scheduling | Courses | Prerequisites |
| Build systems (Make, npm) | Tasks/packages | Dependencies |
| Spreadsheet formulas | Cells | Cell references |
| Compilation order | Source files | #include dependencies |
| Task scheduling | Tasks | "must complete before" |

**DFS-based Topological Sort (with cycle detection):**

```python
def topological_sort(self) -> list[str]:
    visited = set()
    rec_stack = set()      # Track current recursion path for cycle detection
    result = []

    for vertex in self.get_vertices():
        if vertex not in visited:
            if not self._topological_sort_dfs(vertex, visited, rec_stack, result):
                return []      # Cycle detected

    return result[::-1]        # Reverse for correct order

def _topological_sort_dfs(
    self, vertex: str, visited: set, rec_stack: set, result: list
) -> bool:
    """Helper for topological sort. Returns False if cycle detected."""
    visited.add(vertex)
    rec_stack.add(vertex)

    for neighbor in self.get_neighbors(vertex):
        if neighbor not in visited:
            if not self._topological_sort_dfs(neighbor, visited, rec_stack, result):
                return False
        elif neighbor in rec_stack:
            return False       # Cycle detected!

    rec_stack.remove(vertex)
    result.append(vertex)      # Add AFTER processing all neighbors
    return True
```

**Why add vertex AFTER processing neighbors?**

```
DAG: A → B → C

DFS from A:
  Visit A → Visit B → Visit C
  C has no neighbors → add C to result
  B done processing → add B to result
  A done processing → add A to result

result = [C, B, A]
reversed = [A, B, C]  ← Correct topological order!
```

**Time Complexity:**

| Implementation | Time | Why |
|----------------|------|-----|
| Adjacency Matrix | O(V²) | get_neighbors is O(V) - must scan entire row |
| Adjacency List | O(V + E) | get_neighbors is O(degree) - only existing edges |

**Both implementations use identical logic:**

```python
# GraphAdjMatrix - O(V²)
def topological_sort(self) -> list[str]:
    ...
    for vertex in self.get_vertices():        # O(V)
        if vertex not in visited:
            self._topological_sort_dfs(...)   # Each call: O(V) for get_neighbors

def _topological_sort_dfs(self, vertex, ...):
    for neighbor in self.get_neighbors(vertex):  # O(V) - scans entire row
        ...

# GraphAdjList - O(V + E)
def topological_sort(self) -> list[str]:
    ...
    for vertex in self.get_vertices():        # O(V)
        if vertex not in visited:
            self._topological_sort_dfs(...)   # Each call: O(degree) for get_neighbors

def _topological_sort_dfs(self, vertex, ...):
    for neighbor in self.get_neighbors(vertex):  # O(degree) - only existing edges
        ...
```

The code is identical - the time difference comes from `get_neighbors()`:
- **Matrix**: Must check all V slots in the row → O(V) per vertex → O(V²) total
- **List**: Only iterates existing neighbors → O(degree) per vertex → O(V + E) total

**Usage:**

```python
from graphs.graph import GraphAdjList

# Create a DAG
dag = GraphAdjList(directed=True)
dag.add_edge("Math", "Physics")
dag.add_edge("Math", "Stats")
dag.add_edge("Physics", "Quantum")
dag.add_edge("Stats", "ML")

print(dag.topological_sort())  # ['Math', 'Stats', 'ML', 'Physics', 'Quantum']

# Cycle detection
cyclic = GraphAdjList(directed=True)
cyclic.add_edge("A", "B")
cyclic.add_edge("B", "C")
cyclic.add_edge("C", "A")      # Creates cycle

print(cyclic.topological_sort())  # [] (empty = cycle detected)
```

**Cycle Detection with has_cycle():**

While `topological_sort()` returns an empty list when a cycle is detected, you can also explicitly check for cycles using `has_cycle()`:

```python
from graphs.graph import GraphAdjList

# DAG (no cycle)
dag = GraphAdjList(directed=True)
dag.add_edge("A", "B")
dag.add_edge("B", "C")
print(dag.has_cycle())  # False

# Directed graph with cycle
cyclic = GraphAdjList(directed=True)
cyclic.add_edge("A", "B")
cyclic.add_edge("B", "C")
cyclic.add_edge("C", "A")  # Creates cycle A → B → C → A
print(cyclic.has_cycle())  # True

# Undirected graph with cycle
undirected_cyclic = GraphAdjList(directed=False)
undirected_cyclic.add_edge("A", "B")
undirected_cyclic.add_edge("B", "C")
undirected_cyclic.add_edge("C", "A")  # Creates cycle A - B - C - A
print(undirected_cyclic.has_cycle())  # True

# Undirected tree (no cycle)
tree = GraphAdjList(directed=False)
tree.add_edge("A", "B")
tree.add_edge("A", "C")
tree.add_edge("C", "D")
print(tree.has_cycle())  # False
```

**has_cycle() vs topological_sort() for cycle detection (directed graphs only):**

| Method | Returns | Use When |
|--------|---------|----------|
| `has_cycle()` | `bool` | Only need to know if cycle exists (works for both directed and undirected) |
| `topological_sort()` | `[]` if cycle | Need ordering AND cycle check (directed graphs only) |

For directed graphs, both use DFS with a recursion stack. `has_cycle()` is more explicit when you only care about cycle existence without needing the topological order.

**Why topological_sort() doesn't call has_cycle() internally:**

It may seem logical to call `has_cycle()` first, then build the topological order. However, this is less efficient:

| Approach | Traversals | Time |
|----------|------------|------|
| Separate: `has_cycle()` then sort | 2 passes | 2 × O(V + E) |
| Integrated (current) | 1 pass | 1 × O(V + E) |

```python
# Less efficient - two traversals
def topological_sort(self):
    if self.has_cycle():      # First traversal
        return []
    # ... build order ...     # Second traversal

# More efficient - single traversal (current implementation)
def topological_sort(self):
    # Detect cycle AND build order in same DFS
    for neighbor in self.get_neighbors(vertex):
        if neighbor not in visited:
            if not self._topological_sort_dfs(...):
                return False  # Propagate cycle detection upward
        elif neighbor in rec_stack:
            return False      # Cycle detected
```

The integrated approach is better because:
- **Single traversal** - does both cycle detection and ordering in one DFS pass
- **Early termination** - stops immediately when cycle found
- **No duplication** - both use the same DFS with `rec_stack` logic

**Cycle Detection for Directed vs Undirected Graphs:**

`has_cycle()` uses different algorithms depending on graph type:

| Graph Type | Algorithm | How it detects cycles |
|------------|-----------|----------------------|
| Directed | Recursion stack | Back edge = neighbor in current recursion path |
| Undirected | Parent tracking | Visited neighbor ≠ parent = cycle |

**Why undirected needs a different algorithm:**

In undirected graphs, edge A—B means both A→B and B→A. The recursion stack approach would falsely detect cycles:

```
Undirected: A --- B (no cycle)

Recursion stack approach (wrong):
  1. Visit A, rec_stack = {A}
  2. Go to B, rec_stack = {A, B}
  3. B's neighbor is A, A is in rec_stack
  4. FALSE POSITIVE: Reports cycle!

Parent tracking approach (correct):
  1. Visit A, parent = None
  2. Go to B, parent = A
  3. B's neighbor is A, but A == parent
  4. Skip (this is just the edge we came from)
  5. No cycle detected ✓
```

```python
# Directed: recursion stack
def _has_cycle_directed_dfs(self, vertex, visited, rec_stack):
    visited.add(vertex)
    rec_stack.add(vertex)
    for neighbor in self.get_neighbors(vertex):
        if neighbor not in visited:
            if self._has_cycle_directed_dfs(neighbor, visited, rec_stack):
                return True
        elif neighbor in rec_stack:
            return True  # Back edge = cycle
    rec_stack.remove(vertex)
    return False

# Undirected: parent tracking
def _has_cycle_undirected_dfs(self, vertex, visited, parent):
    visited.add(vertex)
    for neighbor in self.get_neighbors(vertex):
        if neighbor not in visited:
            if self._has_cycle_undirected_dfs(neighbor, visited, vertex):
                return True
        elif neighbor != parent:
            return True  # Visited neighbor that's not parent = cycle
    return False
```

**Key differences between the two algorithms:**

| Aspect | Directed | Undirected |
|--------|----------|------------|
| Tracking | `rec_stack` (set of vertices in current path) | `parent` (single vertex we came from) |
| Cycle condition | `neighbor in rec_stack` | `neighbor != parent` |
| Backtracking | Must remove from `rec_stack` | Not needed |
| Memory | O(V) for rec_stack | O(1) for parent |

**Why directed graphs need `rec_stack` instead of just `parent`:**

```
Directed graph:
A → B → D
↓       ↑
C ------+

DFS: A → B → D (backtrack) → A → C → D
```

| Step | vertex | rec_stack | visited | Action |
|------|--------|-----------|---------|--------|
| 1 | A | {A} | {A} | Visit A |
| 2 | B | {A, B} | {A, B} | Visit B |
| 3 | D | {A, B, D} | {A, B, D} | Visit D, no unvisited neighbors |
| 4 | backtrack | {A, B} | {A, B, D} | Remove D from rec_stack |
| 5 | backtrack | {A} | {A, B, D} | Remove B from rec_stack |
| 6 | C | {A, C} | {A, B, D, C} | Visit C |
| 7 | C→D | | | D visited but NOT in rec_stack → No cycle |

If we only tracked `parent` (like undirected), step 7 would see D is visited and D ≠ parent(A), incorrectly reporting a cycle. The `rec_stack` correctly identifies that D is not in the current DFS path.

**Why undirected graphs only need `parent`:**

In undirected graphs, edges are bidirectional. If we reach a visited vertex that's not our immediate parent, it's always a cycle because:
- There's only one path to each vertex in a tree (acyclic graph)
- Finding another path to a visited vertex = cycle

```
Undirected:  A --- B --- C
                   |
                   D

DFS from A: A → B → C (backtrack) → B → D
- At C: neighbor B is visited, but B == parent → Skip (not a cycle)
- At D: neighbor B is visited, but B == parent → Skip (not a cycle)
- No cycle found ✓
```

#### 8.3 Shortest Path (Dijkstra's Algorithm)

Dijkstra's algorithm finds the shortest path between vertices in a weighted graph with **non-negative edge weights**.

```
Weighted Graph:
    A --4-- B --5-- D
    |       |       |
    2       1       2
    |       |       |
    C ------+------ E
        8       10

Shortest path from A to E:
  A → C (2) → B (1) → D (5) → E (2) = 10
  Path: ['A', 'C', 'B', 'D', 'E']
```

**How Dijkstra's Algorithm Works:**

1. Initialize all distances to infinity, except source (distance = 0)
2. Use a priority queue (min-heap) to always process the closest unvisited vertex
3. For each neighbor, check if going through current vertex gives a shorter path
4. Update distance and track the previous vertex if shorter path found
5. Repeat until destination is reached or all reachable vertices are visited

```python
def shortest_path(self, from_vertex: str, to_vertex: str):
    distances = {v: float("inf") for v in vertices}
    distances[from_vertex] = 0
    previous = {v: None for v in vertices}
    pq = [(0, from_vertex)]  # (distance, vertex)
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        if current == to_vertex:
            break  # Found shortest path

        for neighbor, weight in get_neighbors_with_weights(current):
            if neighbor not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

    # Reconstruct path from previous pointers
    path = []
    current = to_vertex
    while current is not None:
        path.append(current)
        current = previous[current]
    return distances[to_vertex], path[::-1]
```

**Early Termination Optimization:**

The `if current == to_vertex: break` is an optimization. When we pop a vertex from the priority queue and it's not yet visited, we have the **guaranteed shortest path** to that vertex.

```python
if current == to_vertex:
    break  # Found shortest path to destination, stop!
```

| Approach | When it stops | Use case |
|----------|---------------|----------|
| With `break` | When destination found | Single destination query |
| Without `break` | When all vertices processed | Find shortest paths to ALL vertices |

```
Example: Graph A → B → C → D → E → ... → Z

shortest_path("A", "C"):
  - With break: Stops after processing C (3 vertices)
  - Without break: Processes all 26 vertices
```

The algorithm is correct without this check, just slower for single-destination queries.

**Time Complexity:**

| Implementation | Time | Why |
|----------------|------|-----|
| Adjacency Matrix | O(V² log V) | O(V²) to check all edges, O(log V) per heap operation |
| Adjacency List | O((V + E) log V) | O(V + E) to visit edges, O(log V) per heap operation |

**Why use a priority queue (min-heap)?**

Dijkstra's algorithm repeatedly needs to find the unvisited vertex with minimum distance:

```python
# With priority queue (current implementation)
current = heapq.heappop(pq)       # O(log V) to extract min

# With linear search
min_dist = float("inf")
for v in unvisited:               # O(V) scan every time
    if distances[v] < min_dist:
        min_dist = distances[v]
        current = v
```

**Time Complexity Breakdown:**

| Component | Linear Search | Priority Queue |
|-----------|---------------|----------------|
| Find minimum V times | O(V) × V = O(V²) | O(log V) × V = O(V log V) |
| Edge relaxation (Matrix) | O(V²) | O(V² log V)* |
| Edge relaxation (List) | O(E) | O(E log V)* |

*Each edge relaxation may push to heap at O(log V) cost.

**How to Calculate Total Time Complexity:**

Total = Find Minimum + Edge Relaxation (take dominant term)

*Linear Search:*
```
Matrix: O(V²) + O(V²) = O(2V²) = O(V²)
List:   O(V²) + O(E)  = O(V² + E) ≈ O(V²)  # V² dominates
```

*Priority Queue:*
```
Matrix: O(V log V) + O(V² log V) = O(V² log V)      # V² log V dominates
List:   O(V log V) + O(E log V)  = O((V + E) log V) # Factor out log V
```

**Total Time Complexity:**

| Implementation | Linear Search | Priority Queue |
|----------------|---------------|----------------|
| Adjacency Matrix | O(V²) | O(V² log V) |
| Adjacency List | O(V² + E) ≈ O(V²) | O((V + E) log V) |

**When is linear search better?**

For **dense graphs** (E ≈ V²):
- Linear search: O(V²)
- Priority queue: O(V² log V)

Linear search avoids the log V factor, so for very dense graphs it can be faster in practice. However, for **sparse graphs** (E << V²), priority queue is significantly better: O((V + E) log V) vs O(V²).

**Current implementation uses priority queue** because most real-world graphs are sparse, and the O((V + E) log V) complexity is generally preferred.

**Limitations:**

- Only works with **non-negative edge weights**
- For negative weights, use Bellman-Ford algorithm
- For unweighted graphs, BFS is simpler and equally efficient

**Usage:**

```python
from graphs.graph import GraphAdjList

graph = GraphAdjList(directed=False)
graph.add_edge("A", "B", 4)
graph.add_edge("A", "C", 2)
graph.add_edge("B", "C", 1)
graph.add_edge("B", "D", 5)
graph.add_edge("C", "D", 8)
graph.add_edge("D", "E", 2)

distance, path = graph.shortest_path("A", "E")
print(f"Distance: {distance}")  # 10
print(f"Path: {path}")          # ['A', 'C', 'B', 'D', 'E']

# Just get distance
print(graph.shortest_distance("A", "D"))  # 8
```

#### 8.4 Minimum Spanning Tree (Prim's Algorithm)

A **Minimum Spanning Tree (MST)** is a subset of edges that:
- Connects all vertices
- Has no cycles (forms a tree)
- Has minimum total edge weight

```
Original Weighted Graph:          MST (total weight = 6):

    A --4-- B                       A       B
    |\     /|                       |      /
    2  \1/  5                       2    1
    |   X   |                       |  /
    C --3-- D                       C --3-- D

Edges: A-B(4), A-C(2),             MST edges: A-C(2), C-B(1), C-D(3)
       B-C(1), B-D(5), C-D(3)      Total: 2 + 1 + 3 = 6
```

**How Prim's Algorithm Works:**

1. Start from any vertex, mark it as visited
2. Add all edges from visited vertices to the priority queue
3. Pick the minimum weight edge that connects to an unvisited vertex
4. Add that vertex to visited and its edge to MST
5. Repeat until all vertices are visited

```python
def minimum_spanning_tree(self, start_vertex=None):
    visited = set()
    mst_edges = []
    total_weight = 0

    # Priority queue: (weight, from_vertex, to_vertex)
    pq = [(0, start_vertex, start_vertex)]

    while pq and len(visited) < vertex_count:
        weight, from_v, to_v = heapq.heappop(pq)

        if to_v in visited:
            continue
        visited.add(to_v)

        # Add edge to MST (skip starting vertex's dummy edge)
        if from_v != to_v:
            mst_edges.append((from_v, to_v, weight))
            total_weight += weight

        # Add edges to unvisited neighbors
        for neighbor, edge_weight in get_neighbors_with_weights(to_v):
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, to_v, neighbor))

    return total_weight, mst_edges
```

**Step-by-step example:**

```
Graph: A--4--B, A--2--C, B--1--C, B--5--D, C--3--D

Step  | Pop from heap      | visited   | MST edges              | Add to heap
------|--------------------|-----------|-----------------------|-------------
1     | (0, A, A)          | {A}       | []                    | (4,A,B), (2,A,C)
2     | (2, A, C)          | {A,C}     | [(A,C,2)]             | (1,C,B), (3,C,D)
3     | (1, C, B)          | {A,C,B}   | [(A,C,2),(C,B,1)]     | (5,B,D)
4     | (3, C, D)          | {A,C,B,D} | [(A,C,2),(C,B,1),(C,D,3)] | -
Done! Total weight = 6
```

**Time Complexity:**

| Implementation | Time | Same as Dijkstra |
|----------------|------|------------------|
| Adjacency Matrix | O(V² log V) | Yes |
| Adjacency List | O((V + E) log V) | Yes |

**MST vs Shortest Path:**

| Aspect | MST (Prim's) | Shortest Path (Dijkstra's) |
|--------|--------------|---------------------------|
| Goal | Connect all vertices with min total weight | Find min distance between two vertices |
| Output | Set of edges (tree) | Single path |
| Considers | Edge weights only | Cumulative path distance |
| Graph type | Undirected only | Directed or undirected |

**Use cases:**
- Network design (minimum cable to connect all computers)
- Circuit design
- Cluster analysis
- Approximation algorithms for NP-hard problems

**Usage:**

```python
from graphs.graph import GraphAdjList

graph = GraphAdjList(directed=False)
graph.add_edge("A", "B", 4)
graph.add_edge("A", "C", 2)
graph.add_edge("B", "C", 1)
graph.add_edge("B", "D", 5)
graph.add_edge("C", "D", 3)

total_weight, edges = graph.minimum_spanning_tree()
print(f"Total weight: {total_weight}")  # 6
print(f"MST edges: {edges}")            # [('A', 'C', 2), ('C', 'B', 1), ('C', 'D', 3)]
```

**Note:** MST only works for **undirected, connected graphs**. Returns `(0, [])` for directed graphs or disconnected graphs.

#### 8.5 Implementation Comparison

**Storage Visualization:**

```
Graph with 4 vertices, 3 edges:

    A --- B
    |
    C --- D

Adjacency Matrix (16 cells, mostly empty):
      A  B  C  D
   A [0, 1, 1, 0]     ← 4 cells per vertex
   B [1, 0, 0, 0]        regardless of edges
   C [1, 0, 0, 1]
   D [0, 0, 1, 0]

Adjacency List (only stores existing edges):
   A: [B, C]          ← only 2 entries
   B: [A]             ← only 1 entry
   C: [A, D]          ← only 2 entries
   D: [C]             ← only 1 entry
```

**Space Complexity:**

| Implementation | Space | Why |
|----------------|-------|-----|
| Matrix | O(V²) | Always V×V array, regardless of edges |
| List | O(V + E) | Only stores existing vertices and edges |

**Dense vs Sparse Graphs:**

```
Dense Graph (E ≈ V²):           Sparse Graph (E << V²):
Many connections                Few connections

  A---B---C                       A     B
  |\ /|\ /|                       |
  | X | X |                       C --- D --- E
  |/ \|/ \|                             |
  D---E---F                             F

Matrix efficient:               List efficient:
- Few wasted cells              - Matrix would waste space
- O(1) edge check useful        - O(V + E) << O(V²)
```

**When to use which:**

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| Dense graph (E ≈ V²) | `GraphAdjMatrix` | Little wasted space, O(1) edge lookup |
| Sparse graph (E << V²) | `GraphAdjList` | Memory efficient, O(V + E) traversal |
| Frequent edge existence checks | `GraphAdjMatrix` | O(1) vs O(degree) |
| Frequent neighbor iteration | `GraphAdjList` | O(degree) vs O(V) to get neighbors |
| Small number of vertices | `GraphAdjMatrix` | Simple, fast edge operations |
| Large graph, few edges | `GraphAdjList` | Memory savings significant |
| Social network (sparse) | `GraphAdjList` | Millions of users, few connections each |
| Flight routes (dense) | `GraphAdjMatrix` | Many airports, most have direct flights |

**Usage:**

```python
from graphs.graph import GraphAdjMatrix, GraphAdjList

# Adjacency List (recommended for most cases)
graph = GraphAdjList(directed=False)
graph.add_edge("A", "B")
graph.add_edge("A", "C")
graph.add_edge("B", "D")
graph.add_edge("C", "D")

print(graph.get_neighbors("A"))      # ['B', 'C']
print(graph.bfs("A"))                # ['A', 'B', 'C', 'D']
print(graph.dfs("A"))                # ['A', 'B', 'D', 'C']
print(graph.has_path("A", "D"))      # True

# Adjacency Matrix
matrix = GraphAdjMatrix(directed=False)
for v in ["A", "B", "C", "D"]:
    matrix.add_vertex(v)
matrix.add_edge("A", "B")
matrix.add_edge("B", "C")
print(matrix.has_edge("A", "B"))     # True
print(matrix.has_edge("A", "C"))     # False

# Directed graph
digraph = GraphAdjList(directed=True)
digraph.add_edge("A", "B")           # A → B
print(digraph.has_edge("A", "B"))    # True
print(digraph.has_edge("B", "A"))    # False

# Weighted graph
weighted = GraphAdjList(directed=False)
weighted.add_edge("A", "B", weight=5)
weighted.add_edge("B", "C", weight=3)
print(weighted.get_edge_weight("A", "B"))  # 5
print(weighted.get_neighbors_with_weights("A"))  # [('B', 5)]
```

#### 8.6 Why No Node/Edge Classes?

A common question is whether graphs should use dedicated `Node` and `Edge` classes instead of the current approach.

**Current approach:**

```python
# Adjacency List: dict of (neighbor, weight) tuples
self.adj_list: dict[str, list[tuple[str, float]]] = {}

# Adjacency Matrix: 2D list with vertex mappings
self.matrix: list[list[float]] = []
self.vertex_map: dict[str, int] = {}
```

**Alternative with Node/Edge classes:**

```python
class Node:
    def __init__(self, label: str):
        self.label = label
        self.data = None      # Additional metadata

class Edge:
    def __init__(self, from_node: Node, to_node: Node, weight: float = 1):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.label = None     # Edge metadata
```

**Comparison:**

| Aspect | Current Approach | Node/Edge Classes |
|--------|------------------|-------------------|
| Simplicity | Simpler, less code | More boilerplate |
| Memory | Less overhead | Object overhead per node/edge |
| Performance | Faster (direct access) | Slightly slower (object indirection) |
| Extensibility | Limited | Easy to add attributes |
| Node metadata | Not supported | `node.color`, `node.visited`, etc. |
| Edge metadata | Only weight | `edge.label`, `edge.capacity`, etc. |
| Textbook alignment | Matches DSA teaching | More software engineering style |

**When Node/Edge classes are better:**

| Scenario | Why |
|----------|-----|
| Need node metadata | Store color, state, timestamps on nodes |
| Need edge metadata | Store labels, capacity (flow networks) |
| Building reusable library | More extensible and maintainable |
| Multigraphs | Multiple edges between same nodes |
| Complex algorithms | Network flow, graph coloring |

**When current approach is better:**

| Scenario | Why |
|----------|-----|
| Educational/learning | Directly mirrors textbook adjacency matrix/list |
| Simple algorithms | BFS, DFS, shortest path don't need metadata |
| Performance-critical | Less object overhead, faster access |
| Vertices are just labels | No need for node objects |

**Why this repo uses the current approach:**

1. **Textbook alignment** - Directly mirrors how adjacency matrix/list are taught in DSA courses
2. **Focus on algorithms** - Emphasizes the algorithm logic, not OOP design patterns
3. **Simpler to understand** - Less abstraction for learning purposes
4. **Less code** - Easier to read and maintain

For production graph libraries or complex applications requiring metadata, Node/Edge classes would be the better choice.

## Running Tests

```bash
cd src
python main.py
```

## Requirements

- Python 3.x
