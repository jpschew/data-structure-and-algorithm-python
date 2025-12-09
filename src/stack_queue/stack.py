from typing import Any, Optional


class Node:
    """A node for the stack."""

    def __init__(self, data: Any) -> None:
        self.data: Any = data
        self.next: Optional[Node] = None


class Stack:
    """Stack implementation using linked list (LIFO - Last In First Out)."""

    def __init__(self) -> None:
        self.top: Optional[Node] = None
        self.size: int = 0

    def is_empty(self) -> bool:
        """Check if stack is empty. O(1)"""
        return self.top is None

    def push(self, data: Any) -> None:
        """Add element to the top of stack. O(1)"""
        new_node = Node(data)
        new_node.next = self.top
        self.top = new_node
        self.size += 1

    def pop(self) -> Optional[Any]:
        """Remove and return element from top of stack. O(1)"""
        if self.top is None:
            return None
        data = self.top.data
        self.top = self.top.next
        self.size -= 1
        return data

    def peek(self) -> Optional[Any]:
        """Return top element without removing it. O(1)"""
        if self.top is None:
            return None
        return self.top.data

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        values = []
        current = self.top
        while current:
            values.append(str(current.data))
            current = current.next
        return " -> ".join(values) if values else "Empty"


def test_stack():
    print("=" * 50)
    print("Testing Stack")
    print("=" * 50)

    stack = Stack()

    # Test empty stack
    print(f"\n1. Empty stack: {stack}")
    print(f"   is_empty(): {stack.is_empty()}")
    print(f"   peek(): {stack.peek()}")
    print(f"   pop(): {stack.pop()}")

    # Test push
    print("\n2. Testing push():")
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print(f"   After pushing 1, 2, 3: {stack}")
    print(f"   Size: {len(stack)}")

    # Test peek
    print("\n3. Testing peek():")
    print(f"   peek(): {stack.peek()}")
    print(f"   Stack unchanged: {stack}")

    # Test pop
    print("\n4. Testing pop():")
    print(f"   pop(): {stack.pop()}")
    print(f"   After pop: {stack}")
    print(f"   pop(): {stack.pop()}")
    print(f"   After pop: {stack}")

    # Test pop until empty
    print("\n5. Testing pop until empty:")
    print(f"   pop(): {stack.pop()}")
    print(f"   After pop: {stack}")
    print(f"   is_empty(): {stack.is_empty()}")
    print(f"   pop() on empty: {stack.pop()}")
