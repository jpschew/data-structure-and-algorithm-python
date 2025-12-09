class Node:
    """A node for the queue."""

    def __init__(self, data):
        self.data = data
        self.next = None


class Queue:
    """Queue implementation using linked list (FIFO - First In First Out)."""

    def __init__(self):
        self.front = None
        self.rear = None
        self.size = 0

    def is_empty(self):
        """Check if queue is empty. O(1)"""
        return self.front is None

    def enqueue(self, data):
        """Add element to the rear of queue. O(1)"""
        new_node = Node(data)
        if self.rear is None:
            self.front = new_node
            self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        self.size += 1

    def dequeue(self):
        """Remove and return element from front of queue. O(1)"""
        if self.front is None:
            return None
        data = self.front.data
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        self.size -= 1
        return data

    def peek(self):
        """Return front element without removing it. O(1)"""
        if self.front is None:
            return None
        return self.front.data

    def __len__(self):
        return self.size

    def __str__(self):
        values = []
        current = self.front
        while current:
            values.append(str(current.data))
            current = current.next
        return " <- ".join(values) if values else "Empty"


# =============================================================================
# Queue Implementation using 2 Stacks
# =============================================================================


class QueueUsingStacks:
    """
    Queue implementation using two stacks (FIFO - First In First Out).

    Approach:
    - stack_in: for enqueue operations
    - stack_out: for dequeue operations

    When dequeue is called and stack_out is empty,
    transfer all elements from stack_in to stack_out (reverses order).

    Time Complexity:
    - enqueue(): O(1)
    - dequeue(): Amortized O(1), worst case O(n)
    - peek(): Amortized O(1), worst case O(n)
    """

    def __init__(self):
        from .stack import Stack
        self.stack_in = Stack()   # For pushing new elements
        self.stack_out = Stack()  # For popping elements

    def is_empty(self):
        """Check if queue is empty. O(1)"""
        return self.stack_in.is_empty() and self.stack_out.is_empty()

    def enqueue(self, data):
        """Add element to the rear of queue. O(1)"""
        self.stack_in.push(data)

    def _transfer(self):
        """Transfer elements from stack_in to stack_out. O(n)"""
        while not self.stack_in.is_empty():
            self.stack_out.push(self.stack_in.pop())

    def dequeue(self):
        """Remove and return element from front of queue. Amortized O(1)"""
        if self.is_empty():
            return None
        if self.stack_out.is_empty():
            self._transfer()
        return self.stack_out.pop()

    def peek(self):
        """Return front element without removing it. Amortized O(1)"""
        if self.is_empty():
            return None
        if self.stack_out.is_empty():
            self._transfer()
        return self.stack_out.peek()

    def __len__(self):
        return len(self.stack_in) + len(self.stack_out)

    def __str__(self):
        # Combine stack_out (reversed) + stack_in for display
        if self.is_empty():
            return "Empty"
        # Collect elements from stack_out (front of queue)
        out_elements = []
        current = self.stack_out.top
        while current:
            out_elements.append(current.data)
            current = current.next
        # Collect elements from stack_in (back of queue, reversed)
        in_elements = []
        current = self.stack_in.top
        while current:
            in_elements.append(current.data)
            current = current.next
        in_elements.reverse()
        elements = out_elements + in_elements
        return " <- ".join(str(x) for x in elements)


def test_queue():
    print("=" * 50)
    print("Testing Queue")
    print("=" * 50)

    queue = Queue()

    # Test empty queue
    print(f"\n1. Empty queue: {queue}")
    print(f"   is_empty(): {queue.is_empty()}")
    print(f"   peek(): {queue.peek()}")
    print(f"   dequeue(): {queue.dequeue()}")

    # Test enqueue
    print("\n2. Testing enqueue():")
    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)
    print(f"   After enqueuing 1, 2, 3: {queue}")
    print(f"   Size: {len(queue)}")
    print(f"   Front: {queue.front.data}, Rear: {queue.rear.data}")

    # Test peek
    print("\n3. Testing peek():")
    print(f"   peek(): {queue.peek()}")
    print(f"   Queue unchanged: {queue}")

    # Test dequeue
    print("\n4. Testing dequeue():")
    print(f"   dequeue(): {queue.dequeue()}")
    print(f"   After dequeue: {queue}")
    print(f"   dequeue(): {queue.dequeue()}")
    print(f"   After dequeue: {queue}")

    # Test dequeue until empty
    print("\n5. Testing dequeue until empty:")
    print(f"   dequeue(): {queue.dequeue()}")
    print(f"   After dequeue: {queue}")
    print(f"   is_empty(): {queue.is_empty()}")
    print(f"   front: {queue.front}, rear: {queue.rear}")
    print(f"   dequeue() on empty: {queue.dequeue()}")


def test_queue_using_stacks():
    print("\n" + "=" * 50)
    print("Testing QueueUsingStacks")
    print("=" * 50)

    queue = QueueUsingStacks()

    # Test empty queue
    print(f"\n1. Empty queue: {queue}")
    print(f"   is_empty(): {queue.is_empty()}")
    print(f"   peek(): {queue.peek()}")
    print(f"   dequeue(): {queue.dequeue()}")

    # Test enqueue
    print("\n2. Testing enqueue():")
    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)
    print(f"   After enqueuing 1, 2, 3: {queue}")
    print(f"   Size: {len(queue)}")
    print(f"   stack_in: {queue.stack_in}, stack_out: {queue.stack_out}")

    # Test peek (triggers transfer)
    print("\n3. Testing peek() [triggers transfer]:")
    print(f"   peek(): {queue.peek()}")
    print(f"   stack_in: {queue.stack_in}, stack_out: {queue.stack_out}")
    print(f"   Queue unchanged: {queue}")

    # Test dequeue
    print("\n4. Testing dequeue():")
    print(f"   dequeue(): {queue.dequeue()}")
    print(f"   After dequeue: {queue}")
    print(f"   stack_in: {queue.stack_in}, stack_out: {queue.stack_out}")

    # Add more elements and dequeue
    print("\n5. Testing mixed enqueue/dequeue:")
    queue.enqueue(4)
    queue.enqueue(5)
    print(f"   After enqueue 4, 5: {queue}")
    print(f"   stack_in: {queue.stack_in}, stack_out: {queue.stack_out}")
    print(f"   dequeue(): {queue.dequeue()}")
    print(f"   After dequeue: {queue}")
    print(f"   stack_in: {queue.stack_in}, stack_out: {queue.stack_out}")

    # Test dequeue until empty
    print("\n6. Testing dequeue until empty:")
    while not queue.is_empty():
        print(f"   dequeue(): {queue.dequeue()}")
    print(f"   After all dequeues: {queue}")
    print(f"   is_empty(): {queue.is_empty()}")
    print(f"   dequeue() on empty: {queue.dequeue()}")


# =============================================================================
# Priority Queue Implementations
# Lower value = higher priority (Min-Priority Queue)
# =============================================================================


class PriorityNode:
    """A node for the priority queue with priority field."""

    def __init__(self, data, priority):
        self.data = data
        self.priority = priority
        self.next = None


class PriorityQueueUnsortedList:
    """
    Priority Queue using UNSORTED Python LIST.

    Time Complexity:
    - enqueue(): O(1) - append to end
    - dequeue(): O(n) - scan to find min, O(n) shift after pop
    - peek(): O(n) - scan to find min
    """

    def __init__(self):
        self.items = []  # stores (priority, data) tuples

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, data, priority):
        """Add element with priority. O(1)"""
        self.items.append((priority, data))

    def _find_min_index(self):
        """Find index of minimum priority. O(n)"""
        if self.is_empty():
            return None
        min_index = 0
        for i in range(1, len(self.items)):
            if self.items[i][0] < self.items[min_index][0]:
                min_index = i
        return min_index

    def dequeue(self):
        """Remove and return highest priority element. O(n)"""
        if self.is_empty():
            return None
        min_index = self._find_min_index()
        priority, data = self.items.pop(min_index)  # O(n) shift
        return data

    def peek(self):
        """Return highest priority element. O(n)"""
        if self.is_empty():
            return None
        min_index = self._find_min_index()
        return self.items[min_index][1]

    def __len__(self):
        return len(self.items)

    def __str__(self):
        if self.is_empty():
            return "Empty"
        return ", ".join(f"({p}, {d})" for p, d in self.items)


class PriorityQueueSortedList:
    """
    Priority Queue using SORTED Python LIST.
    Keeps elements sorted by priority (descending - highest priority at end).

    Time Complexity:
    - enqueue(): O(n) - find position + O(n) shift for insert
    - dequeue(): O(1) - pop from end (no shift)
    - peek(): O(1) - look at end
    """

    def __init__(self):
        self.items = []  # stores (priority, data), sorted descending

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, data, priority):
        """Add element in sorted position. O(n)"""
        # Find position (keep descending order so min is at end)
        pos = 0
        for i in range(len(self.items)):
            if self.items[i][0] > priority:
                pos = i + 1
            else:
                break
        self.items.insert(pos, (priority, data))  # O(n) shift

    def dequeue(self):
        """Remove and return highest priority element. O(1)"""
        if self.is_empty():
            return None
        priority, data = self.items.pop()  # O(1) from end
        return data

    def peek(self):
        """Return highest priority element. O(1)"""
        if self.is_empty():
            return None
        return self.items[-1][1]

    def __len__(self):
        return len(self.items)

    def __str__(self):
        if self.is_empty():
            return "Empty"
        return ", ".join(f"({p}, {d})" for p, d in self.items)


class PriorityQueueUnsortedLinkedList:
    """
    Priority Queue using UNSORTED LINKED LIST.

    Time Complexity:
    - enqueue(): O(1) - prepend to head
    - dequeue(): O(n) - scan to find min, O(1) removal
    - peek(): O(n) - scan to find min
    """

    def __init__(self):
        self.head = None
        self.size = 0

    def is_empty(self):
        return self.head is None

    def enqueue(self, data, priority):
        """Add element at head. O(1)"""
        new_node = PriorityNode(data, priority)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def dequeue(self):
        """Remove and return highest priority element. O(n)"""
        if self.head is None:
            return None

        # Find node with minimum priority
        min_node = self.head
        min_prev = None
        current = self.head
        prev = None

        while current:
            if current.priority < min_node.priority:
                min_node = current
                min_prev = prev
            prev = current
            current = current.next

        # Remove min_node
        data = min_node.data
        if min_prev is None:
            self.head = min_node.next
        else:
            min_prev.next = min_node.next
        self.size -= 1
        return data

    def peek(self):
        """Return highest priority element. O(n)"""
        if self.head is None:
            return None
        min_node = self.head
        current = self.head.next
        while current:
            if current.priority < min_node.priority:
                min_node = current
            current = current.next
        return min_node.data

    def __len__(self):
        return self.size

    def __str__(self):
        if self.head is None:
            return "Empty"
        values = []
        current = self.head
        while current:
            values.append(f"({current.priority}, {current.data})")
            current = current.next
        return " -> ".join(values)


class PriorityQueueSortedLinkedList:
    """
    Priority Queue using SORTED LINKED LIST.
    Keeps elements sorted by priority (ascending - highest priority at head).

    Time Complexity:
    - enqueue(): O(n) - find position, O(1) insertion
    - dequeue(): O(1) - remove from head
    - peek(): O(1) - look at head
    """

    def __init__(self):
        self.head = None
        self.size = 0

    def is_empty(self):
        return self.head is None

    def enqueue(self, data, priority):
        """Add element in sorted position. O(n)"""
        new_node = PriorityNode(data, priority)

        # Insert at head if empty or higher priority than head
        if self.head is None or priority < self.head.priority:
            new_node.next = self.head
            self.head = new_node
        else:
            # Find correct position
            current = self.head
            while current.next and current.next.priority <= priority:
                current = current.next
            new_node.next = current.next
            current.next = new_node
        self.size += 1

    def dequeue(self):
        """Remove and return highest priority element. O(1)"""
        if self.head is None:
            return None
        data = self.head.data
        self.head = self.head.next
        self.size -= 1
        return data

    def peek(self):
        """Return highest priority element. O(1)"""
        if self.head is None:
            return None
        return self.head.data

    def __len__(self):
        return self.size

    def __str__(self):
        if self.head is None:
            return "Empty"
        values = []
        current = self.head
        while current:
            values.append(f"({current.priority}, {current.data})")
            current = current.next
        return " -> ".join(values)


def test_priority_queues():
    print("\n" + "=" * 60)
    print("Testing All Priority Queue Implementations")
    print("=" * 60)

    # Test data: (data, priority)
    test_data = [("Low", 3), ("High", 1), ("Medium", 2), ("Urgent", 0)]

    # Test all 4 implementations
    implementations = [
        ("PriorityQueueUnsortedList", PriorityQueueUnsortedList()),
        ("PriorityQueueSortedList", PriorityQueueSortedList()),
        ("PriorityQueueUnsortedLinkedList", PriorityQueueUnsortedLinkedList()),
        ("PriorityQueueSortedLinkedList", PriorityQueueSortedLinkedList()),
    ]

    for name, pq in implementations:
        print(f"\n{'─' * 60}")
        print(f"Testing {name}")
        print(f"{'─' * 60}")

        # Enqueue all items
        print(f"\n1. Enqueue: (3,Low), (1,High), (2,Medium), (0,Urgent)")
        for data, priority in test_data:
            pq.enqueue(data, priority)
        print(f"   Queue: {pq}")

        # Peek
        print(f"\n2. peek(): {pq.peek()}")

        # Dequeue all (should come out in priority order)
        print(f"\n3. Dequeue all (should be: Urgent, High, Medium, Low):")
        while not pq.is_empty():
            print(f"   dequeue(): {pq.dequeue()}")

    # Comparison table
    print(f"\n{'=' * 60}")
    print("Time Complexity Comparison")
    print(f"{'=' * 60}")
    print(f"\n{'Implementation':<35} {'enqueue':<12} {'dequeue':<12} {'peek':<10}")
    print(f"{'-' * 35} {'-' * 12} {'-' * 12} {'-' * 10}")
    print(f"{'Unsorted List':<35} {'O(1)':<12} {'O(n)':<12} {'O(n)':<10}")
    print(f"{'Sorted List':<35} {'O(n)':<12} {'O(1)':<12} {'O(1)':<10}")
    print(f"{'Unsorted Linked List':<35} {'O(1)':<12} {'O(n)':<12} {'O(n)':<10}")
    print(f"{'Sorted Linked List':<35} {'O(n)':<12} {'O(1)':<12} {'O(1)':<10}")
    print(f"\nNote: Sorted List stores in descending order (min at end)")
    print(f"      Sorted Linked List stores in ascending order (min at head)")
