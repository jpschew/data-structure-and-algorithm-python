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
    └── heaps/
        └── binary_heap.py      # MinHeap and MaxHeap implementations
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

## Running Tests

```bash
cd src
python main.py
```

## Requirements

- Python 3.x
