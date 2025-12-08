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
