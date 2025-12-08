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

### 4. Binary Search Tree

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
| inorder() | Left, Root, Right | Sorted output |
| preorder() | Root, Left, Right | Copy tree |
| postorder() | Left, Right, Root | Delete tree |
| level_order() | BFS by level | Level-by-level processing |

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
