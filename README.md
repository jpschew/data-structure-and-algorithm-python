# Data Structures and Algorithms (DSA)

A Python implementation of fundamental data structures with comprehensive test cases.

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
    └── trees/
        └── binary_tree.py      # Binary Search Tree implementation
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

#### Queue Using Two Stacks

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

Min-Priority Queue implementations (lower value = higher priority). Four variants comparing sorted vs unsorted and list vs linked list:

| Implementation | enqueue() | dequeue() | peek() |
|----------------|-----------|-----------|--------|
| `PriorityQueueUnsortedList` | O(1) | O(n) | O(n) |
| `PriorityQueueSortedList` | O(n) | O(1) | O(1) |
| `PriorityQueueUnsortedLinkedList` | O(1) | O(n) | O(n) |
| `PriorityQueueSortedLinkedList` | O(n) | O(1) | O(1) |
| Heap (Binary Heap) | O(log n) | O(log n) | O(1) |

*Note: Heap-based priority queue will be covered when heaps data structure is introduced.*

**Trade-offs:**
- **Unsorted**: Fast insert, slow removal (good for many inserts, few removals)
- **Sorted**: Slow insert, fast removal (good for few inserts, many removals)

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

### 5. Trees

#### What is a Binary Tree?

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

#### Binary Search Tree

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

## Running Tests

```bash
cd src
python main.py
```

## Requirements

- Python 3.x
