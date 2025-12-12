from typing import Any, Optional


class MinHeap:
    """
    Min-Heap implementation using array (smallest element at root).

    Heap Property: parent <= children

    Array representation of complete binary tree:
    - Parent of index i: (i - 1) // 2
    - Left child of index i: 2 * i + 1
    - Right child of index i: 2 * i + 2

    Time Complexity:
    - insert(): O(log n) - heapify up
    - extract_min(): O(log n) - heapify down
    - peek(): O(1)
    - heapify (build heap): O(n)
    """

    def __init__(self) -> None:
        self.heap: list[Any] = []

    def is_empty(self) -> bool:
        """Check if heap is empty. O(1)"""
        return len(self.heap) == 0

    def _parent(self, index: int) -> int:
        """Get parent index."""
        return (index - 1) // 2

    def _left_child(self, index: int) -> int:
        """Get left child index."""
        return 2 * index + 1

    def _right_child(self, index: int) -> int:
        """Get right child index."""
        return 2 * index + 2

    def _has_left_child(self, index: int) -> bool:
        """Check if node has left child."""
        return self._left_child(index) < len(self.heap)

    def _has_right_child(self, index: int) -> bool:
        """Check if node has right child."""
        return self._right_child(index) < len(self.heap)

    def _has_parent(self, index: int) -> bool:
        """Check if node has parent."""
        return self._parent(index) >= 0

    def _swap(self, i: int, j: int) -> None:
        """Swap two elements in heap."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heapify_up(self, index: int) -> None:
        """
        Move element up to maintain heap property.
        Used after insertion.
        """
        while self._has_parent(index) and self.heap[self._parent(index)] > self.heap[index]:
            parent_index = self._parent(index)
            self._swap(parent_index, index)
            index = parent_index

    def _heapify_down(self, index: int) -> None:
        """
        Move element down to maintain heap property.
        Used after extraction.
        """
        while self._has_left_child(index):
            # Find smaller child
            smaller_child_index = self._left_child(index)
            if self._has_right_child(index) and self.heap[self._right_child(index)] < self.heap[smaller_child_index]:
                smaller_child_index = self._right_child(index)

            # If heap property satisfied, stop
            if self.heap[index] <= self.heap[smaller_child_index]:
                break

            # Swap with smaller child
            self._swap(index, smaller_child_index)
            index = smaller_child_index

    def insert(self, data: Any) -> None:
        """Insert element into heap. O(log n)"""
        self.heap.append(data)
        self._heapify_up(len(self.heap) - 1)

    def extract_min(self) -> Optional[Any]:
        """Remove and return minimum element. O(log n)"""
        if self.is_empty():
            return None

        min_val = self.heap[0]

        # Move last element to root and heapify down
        self.heap[0] = self.heap[-1]
        self.heap.pop()

        if not self.is_empty():
            self._heapify_down(0)

        return min_val

    def peek(self) -> Optional[Any]:
        """Return minimum element without removing. O(1)"""
        if self.is_empty():
            return None
        return self.heap[0]

    def heapify(self, array: list[Any]) -> None:
        """
        Build heap from array. O(n)

        More efficient than inserting one by one O(n log n).
        Start from last non-leaf node and heapify down.
        """
        self.heap = array.copy()
        # Start from last non-leaf node
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)

    def heapify_naive(self, array: list[Any]) -> None:
        """
        Build heap from array by inserting one by one. O(n log n)

        This is the naive/unoptimized approach for comparison.
        Each insert is O(log n), so n inserts = O(n log n).

        Compare with heapify() which is O(n).
        """
        self.heap = []
        for item in array:
            self.insert(item)

    def heapify_inplace(self, array: list[Any]) -> None:
        """
        Build heap from array IN-PLACE. O(n) time, O(1) extra space.

        Modifies the original array directly.
        The heap's internal list references the same array.

        Warning: After calling this, modifications to self.heap
        will also modify the original array.
        """
        self.heap = array  # Reference, not copy
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)

    def heap_sort(self, array: list[Any]) -> list[Any]:
        """
        Sort array using heap sort. O(n log n)

        MinHeap heap sort produces ASCENDING order.

        Steps:
        1. Build min heap from array - O(n)
        2. Extract min repeatedly and append to result - O(n log n)

        Note: This creates a new sorted list and clears the heap.
        """
        self.heapify(array)
        result: list[Any] = []
        while not self.is_empty():
            result.append(self.extract_min())
        return result

    def __len__(self) -> int:
        return len(self.heap)

    def __str__(self) -> str:
        if self.is_empty():
            return "Empty"
        return f"MinHeap({self.heap})"


class MaxHeap:
    """
    Max-Heap implementation using array (largest element at root).

    Heap Property: parent >= children

    Time Complexity:
    - insert(): O(log n) - heapify up
    - extract_max(): O(log n) - heapify down
    - peek(): O(1)
    - heapify (build heap): O(n)
    """

    def __init__(self) -> None:
        self.heap: list[Any] = []

    def is_empty(self) -> bool:
        """Check if heap is empty. O(1)"""
        return len(self.heap) == 0

    def _parent(self, index: int) -> int:
        """Get parent index."""
        return (index - 1) // 2

    def _left_child(self, index: int) -> int:
        """Get left child index."""
        return 2 * index + 1

    def _right_child(self, index: int) -> int:
        """Get right child index."""
        return 2 * index + 2

    def _has_left_child(self, index: int) -> bool:
        """Check if node has left child."""
        return self._left_child(index) < len(self.heap)

    def _has_right_child(self, index: int) -> bool:
        """Check if node has right child."""
        return self._right_child(index) < len(self.heap)

    def _has_parent(self, index: int) -> bool:
        """Check if node has parent."""
        return self._parent(index) >= 0

    def _swap(self, i: int, j: int) -> None:
        """Swap two elements in heap."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heapify_up(self, index: int) -> None:
        """
        Move element up to maintain heap property.
        Used after insertion.
        """
        while self._has_parent(index) and self.heap[self._parent(index)] < self.heap[index]:
            parent_index = self._parent(index)
            self._swap(parent_index, index)
            index = parent_index

    def _heapify_down(self, index: int) -> None:
        """
        Move element down to maintain heap property.
        Used after extraction.
        """
        while self._has_left_child(index):
            # Find larger child
            larger_child_index = self._left_child(index)
            if self._has_right_child(index) and self.heap[self._right_child(index)] > self.heap[larger_child_index]:
                larger_child_index = self._right_child(index)

            # If heap property satisfied, stop
            if self.heap[index] >= self.heap[larger_child_index]:
                break

            # Swap with larger child
            self._swap(index, larger_child_index)
            index = larger_child_index

    def insert(self, data: Any) -> None:
        """Insert element into heap. O(log n)"""
        self.heap.append(data)
        self._heapify_up(len(self.heap) - 1)

    def extract_max(self) -> Optional[Any]:
        """Remove and return maximum element. O(log n)"""
        if self.is_empty():
            return None

        max_val = self.heap[0]

        # Move last element to root and heapify down
        self.heap[0] = self.heap[-1]
        self.heap.pop()

        if not self.is_empty():
            self._heapify_down(0)

        return max_val

    def peek(self) -> Optional[Any]:
        """Return maximum element without removing. O(1)"""
        if self.is_empty():
            return None
        return self.heap[0]

    def heapify(self, array: list[Any]) -> None:
        """
        Build heap from array. O(n)

        More efficient than inserting one by one O(n log n).
        Start from last non-leaf node and heapify down.
        """
        self.heap = array.copy()
        # Start from last non-leaf node
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)

    def heapify_naive(self, array: list[Any]) -> None:
        """
        Build heap from array by inserting one by one. O(n log n)

        This is the naive/unoptimized approach for comparison.
        Each insert is O(log n), so n inserts = O(n log n).

        Compare with heapify() which is O(n).
        """
        self.heap = []
        for item in array:
            self.insert(item)

    def heapify_inplace(self, array: list[Any]) -> None:
        """
        Build heap from array IN-PLACE. O(n) time, O(1) extra space.

        Modifies the original array directly.
        The heap's internal list references the same array.

        Warning: After calling this, modifications to self.heap
        will also modify the original array.
        """
        self.heap = array  # Reference, not copy
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self._heapify_down(i)

    def heap_sort(self, array: list[Any]) -> list[Any]:
        """
        Sort array using heap sort. O(n log n)

        MaxHeap heap sort produces DESCENDING order.

        Steps:
        1. Build max heap from array - O(n)
        2. Extract max repeatedly and append to result - O(n log n)

        Note: This creates a new sorted list and clears the heap.
        """
        self.heapify(array)
        result: list[Any] = []
        while not self.is_empty():
            result.append(self.extract_max())
        return result

    def __len__(self) -> int:
        return len(self.heap)

    def __str__(self) -> str:
        if self.is_empty():
            return "Empty"
        return f"MaxHeap({self.heap})"


class PriorityQueueHeap:
    """
    Priority Queue implementation using MinHeap.
    Lower priority value = higher priority (min-priority queue).

    Time Complexity:
    - enqueue(): O(log n)
    - dequeue(): O(log n)
    - peek(): O(1)

    Compared to list-based implementations:
    | Implementation | enqueue | dequeue | peek |
    |----------------|---------|---------|------|
    | Unsorted List  | O(1)    | O(n)    | O(n) |
    | Sorted List    | O(n)    | O(1)    | O(1) |
    | Heap           | O(log n)| O(log n)| O(1) |

    Heap provides balanced performance for both operations.
    """

    def __init__(self) -> None:
        self.heap: list[tuple[int, Any]] = []  # stores (priority, data) tuples

    def is_empty(self) -> bool:
        """Check if priority queue is empty. O(1)"""
        return len(self.heap) == 0

    def _parent(self, index: int) -> int:
        return (index - 1) // 2

    def _left_child(self, index: int) -> int:
        return 2 * index + 1

    def _right_child(self, index: int) -> int:
        return 2 * index + 2

    def _has_left_child(self, index: int) -> bool:
        return self._left_child(index) < len(self.heap)

    def _has_right_child(self, index: int) -> bool:
        return self._right_child(index) < len(self.heap)

    def _has_parent(self, index: int) -> bool:
        return self._parent(index) >= 0

    def _swap(self, i: int, j: int) -> None:
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heapify_up(self, index: int) -> None:
        """Move element up to maintain heap property."""
        while self._has_parent(index) and self.heap[self._parent(index)][0] > self.heap[index][0]:
            parent_index = self._parent(index)
            self._swap(parent_index, index)
            index = parent_index

    def _heapify_down(self, index: int) -> None:
        """Move element down to maintain heap property."""
        while self._has_left_child(index):
            smaller_child_index = self._left_child(index)
            if self._has_right_child(index) and self.heap[self._right_child(index)][0] < self.heap[smaller_child_index][0]:
                smaller_child_index = self._right_child(index)

            if self.heap[index][0] <= self.heap[smaller_child_index][0]:
                break

            self._swap(index, smaller_child_index)
            index = smaller_child_index

    def enqueue(self, data: Any, priority: int) -> None:
        """Add element with priority. O(log n)"""
        self.heap.append((priority, data))
        self._heapify_up(len(self.heap) - 1)

    def dequeue(self) -> Optional[Any]:
        """Remove and return highest priority (lowest value) element. O(log n)"""
        if self.is_empty():
            return None

        priority, data = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()

        if not self.is_empty():
            self._heapify_down(0)

        return data

    def peek(self) -> Optional[Any]:
        """Return highest priority element without removing. O(1)"""
        if self.is_empty():
            return None
        return self.heap[0][1]

    def __len__(self) -> int:
        return len(self.heap)

    def __str__(self) -> str:
        if self.is_empty():
            return "Empty"
        return f"PriorityQueueHeap({[(p, d) for p, d in self.heap]})"


def test_priority_queue_heap():
    print("\n" + "=" * 60)
    print("Testing PriorityQueueHeap")
    print("=" * 60)

    pq = PriorityQueueHeap()

    # Test empty queue
    print(f"\n1. Empty queue: {pq}")
    print(f"   is_empty(): {pq.is_empty()}")
    print(f"   peek(): {pq.peek()}")
    print(f"   dequeue(): {pq.dequeue()}")

    # Test enqueue
    print("\n2. Testing enqueue():")
    test_data = [("Low", 3), ("High", 1), ("Medium", 2), ("Urgent", 0)]
    for data, priority in test_data:
        pq.enqueue(data, priority)
        print(f"   enqueue('{data}', {priority}): {pq}")

    print(f"\n   Size: {len(pq)}")
    print(f"   peek(): {pq.peek()}")

    # Test dequeue
    print("\n3. Testing dequeue() - should come out in priority order:")
    print("   Expected: Urgent (0), High (1), Medium (2), Low (3)")
    while not pq.is_empty():
        print(f"   dequeue(): {pq.dequeue()}")

    # Comparison with list-based implementations
    print("\n" + "=" * 60)
    print("Time Complexity Comparison")
    print("=" * 60)
    print(f"\n{'Implementation':<25} {'enqueue':<12} {'dequeue':<12} {'peek':<10}")
    print(f"{'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10}")
    print(f"{'Unsorted List':<25} {'O(1)':<12} {'O(n)':<12} {'O(n)':<10}")
    print(f"{'Sorted List':<25} {'O(n)':<12} {'O(1)':<12} {'O(1)':<10}")
    print(f"{'Heap (this)':<25} {'O(log n)':<12} {'O(log n)':<12} {'O(1)':<10}")
    print(f"\nHeap provides balanced O(log n) for both insert and remove.")


def test_min_heap():
    print("=" * 60)
    print("Testing MinHeap")
    print("=" * 60)

    heap = MinHeap()

    # Test empty heap
    print(f"\n1. Empty heap: {heap}")
    print(f"   is_empty(): {heap.is_empty()}")
    print(f"   peek(): {heap.peek()}")
    print(f"   extract_min(): {heap.extract_min()}")

    # Test insert
    print("\n2. Testing insert():")
    for val in [5, 3, 8, 1, 2, 9, 4]:
        heap.insert(val)
        print(f"   insert({val}): {heap.heap}")

    print(f"\n   Final heap: {heap}")
    print(f"   Size: {len(heap)}")
    print(f"   Min (peek): {heap.peek()}")

    # Test extract_min
    print("\n3. Testing extract_min():")
    print("   Extracting all elements (should be in ascending order):")
    while not heap.is_empty():
        print(f"   extract_min(): {heap.extract_min()}")

    # Test heapify
    print("\n4. Testing heapify() - O(n) optimized:")
    array = [5, 3, 8, 1, 2, 9, 4]
    print(f"   Input array: {array}")
    heap.heapify(array)
    print(f"   After heapify: {heap.heap}")
    print(f"   Min (peek): {heap.peek()}")

    # Test heapify_naive
    print("\n5. Testing heapify_naive() - O(n log n) unoptimized:")
    array = [5, 3, 8, 1, 2, 9, 4]
    print(f"   Input array: {array}")
    heap.heapify_naive(array)
    print(f"   After heapify_naive: {heap.heap}")
    print(f"   Min (peek): {heap.peek()}")

    # Compare heapify methods
    print("\n6. Comparing heapify() vs heapify_naive():")
    heap1 = MinHeap()
    heap2 = MinHeap()
    heap1.heapify(array)
    heap2.heapify_naive(array)
    print(f"   heapify() result:       {heap1.heap}")
    print(f"   heapify_naive() result: {heap2.heap}")
    print("   Note: Internal structure may differ, but both are valid heaps")

    # Test heapify_inplace
    print("\n7. Testing heapify_inplace() - modifies original array:")
    arr_copy = [5, 3, 8, 1, 2, 9, 4]
    arr_inplace = [5, 3, 8, 1, 2, 9, 4]
    heap_copy = MinHeap()
    heap_inplace = MinHeap()
    heap_copy.heapify(arr_copy)
    heap_inplace.heapify_inplace(arr_inplace)
    print(f"   heapify():        original={arr_copy}, heap={heap_copy.heap}")
    print(f"   heapify_inplace(): original={arr_inplace}, heap={heap_inplace.heap}")
    print(f"   Same object? heapify={arr_copy is heap_copy.heap}, inplace={arr_inplace is heap_inplace.heap}")

    # Test heap_sort
    print("\n8. Testing heap_sort():")
    array = [5, 3, 8, 1, 2, 9, 4]
    print(f"   Input array: {array}")
    sorted_array = heap.heap_sort(array)
    print(f"   MinHeap heap_sort (ascending): {sorted_array}")


def test_max_heap():
    print("\n" + "=" * 60)
    print("Testing MaxHeap")
    print("=" * 60)

    heap = MaxHeap()

    # Test empty heap
    print(f"\n1. Empty heap: {heap}")
    print(f"   is_empty(): {heap.is_empty()}")
    print(f"   peek(): {heap.peek()}")
    print(f"   extract_max(): {heap.extract_max()}")

    # Test insert
    print("\n2. Testing insert():")
    for val in [5, 3, 8, 1, 2, 9, 4]:
        heap.insert(val)
        print(f"   insert({val}): {heap.heap}")

    print(f"\n   Final heap: {heap}")
    print(f"   Size: {len(heap)}")
    print(f"   Max (peek): {heap.peek()}")

    # Test extract_max
    print("\n3. Testing extract_max():")
    print("   Extracting all elements (should be in descending order):")
    while not heap.is_empty():
        print(f"   extract_max(): {heap.extract_max()}")

    # Test heapify
    print("\n4. Testing heapify() - O(n) optimized:")
    array = [5, 3, 8, 1, 2, 9, 4]
    print(f"   Input array: {array}")
    heap.heapify(array)
    print(f"   After heapify: {heap.heap}")
    print(f"   Max (peek): {heap.peek()}")

    # Test heapify_naive
    print("\n5. Testing heapify_naive() - O(n log n) unoptimized:")
    array = [5, 3, 8, 1, 2, 9, 4]
    print(f"   Input array: {array}")
    heap.heapify_naive(array)
    print(f"   After heapify_naive: {heap.heap}")
    print(f"   Max (peek): {heap.peek()}")

    # Compare heapify methods
    print("\n6. Comparing heapify() vs heapify_naive():")
    heap1 = MaxHeap()
    heap2 = MaxHeap()
    heap1.heapify(array)
    heap2.heapify_naive(array)
    print(f"   heapify() result:       {heap1.heap}")
    print(f"   heapify_naive() result: {heap2.heap}")
    print("   Note: Internal structure may differ, but both are valid heaps")

    # Test heapify_inplace
    print("\n7. Testing heapify_inplace() - modifies original array:")
    arr_copy = [5, 3, 8, 1, 2, 9, 4]
    arr_inplace = [5, 3, 8, 1, 2, 9, 4]
    heap_copy = MaxHeap()
    heap_inplace = MaxHeap()
    heap_copy.heapify(arr_copy)
    heap_inplace.heapify_inplace(arr_inplace)
    print(f"   heapify():        original={arr_copy}, heap={heap_copy.heap}")
    print(f"   heapify_inplace(): original={arr_inplace}, heap={heap_inplace.heap}")
    print(f"   Same object? heapify={arr_copy is heap_copy.heap}, inplace={arr_inplace is heap_inplace.heap}")

    # Test heap_sort
    print("\n8. Testing heap_sort():")
    array = [5, 3, 8, 1, 2, 9, 4]
    print(f"   Input array: {array}")
    sorted_array = heap.heap_sort(array)
    print(f"   MaxHeap heap_sort (descending): {sorted_array}")


def test_heap_comparison():
    print("\n" + "=" * 60)
    print("MinHeap vs MaxHeap Comparison")
    print("=" * 60)

    array = [5, 3, 8, 1, 2, 9, 4]
    print(f"\nInput array: {array}")

    min_heap = MinHeap()
    min_heap.heapify(array.copy())

    max_heap = MaxHeap()
    max_heap.heapify(array.copy())

    print(f"\nMinHeap: {min_heap.heap}")
    print(f"MaxHeap: {max_heap.heap}")

    print(f"\nMinHeap root (smallest): {min_heap.peek()}")
    print(f"MaxHeap root (largest): {max_heap.peek()}")

    print("\n" + "=" * 60)
    print("Time Complexity Summary")
    print("=" * 60)
    print(f"\n{'Operation':<20} {'Time':<15} {'Description':<30}")
    print(f"{'-' * 20} {'-' * 15} {'-' * 30}")
    print(f"{'insert()':<20} {'O(log n)':<15} {'Add element, heapify up':<30}")
    print(f"{'extract_min/max()':<20} {'O(log n)':<15} {'Remove root, heapify down':<30}")
    print(f"{'peek()':<20} {'O(1)':<15} {'View root element':<30}")
    print(f"{'heapify()':<20} {'O(n)':<15} {'Build heap from array':<30}")
    print(f"{'heap_sort()':<20} {'O(n log n)':<15} {'Sort array using heap':<30}")
    print(f"{'is_empty()':<20} {'O(1)':<15} {'Check if empty':<30}")
