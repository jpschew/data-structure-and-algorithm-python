from typing import Any, Optional


class Node:
    """A node in a singly linked list."""

    def __init__(self, data: Any) -> None:
        self.data: Any = data
        self.next: Optional[Node] = None


class LinkedListHeadOnly:
    """Singly linked list with head reference only."""

    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.size: int = 0

    def is_empty(self) -> bool:
        return self.head is None

    def prepend(self, data: Any) -> None:
        """Add node at the beginning. O(1)"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def append(self, data: Any) -> None:
        """Add node at the end. O(n) - must traverse to find tail"""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.size += 1
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        self.size += 1

    def insert(self, data: Any, index: int) -> bool:
        """Insert node at specific index. O(n)"""
        if index < 0 or index > self.size:
            return False
        if index == 0:
            self.prepend(data)
            return True
        if index == self.size:
            self.append(data)
            return True
        new_node = Node(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        return True

    def delete(self, data: Any) -> bool:
        """Delete first node with matching data. O(n)"""
        if self.head is None:
            return False
        if self.head.data == data:
            self.head = self.head.next
            self.size -= 1
            return True
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        return False

    def delete_first(self) -> Optional[Any]:
        """Delete node from the beginning. O(1)"""
        if self.head is None:
            return None
        data = self.head.data
        self.head = self.head.next
        self.size -= 1
        return data

    def delete_last(self) -> Optional[Any]:
        """Delete node from the end. O(n) - must traverse to find second-to-last"""
        if self.head is None:
            return None
        if self.head.next is None:
            data = self.head.data
            self.head = None
            self.size -= 1
            return data
        current = self.head
        while current.next.next:
            current = current.next
        data = current.next.data
        current.next = None
        self.size -= 1
        return data

    def delete_at(self, index: int) -> Optional[Any]:
        """Delete node at specific index. O(n)"""
        if index < 0 or index >= self.size:
            return None
        if index == 0:
            return self.delete_first()
        if index == self.size - 1:
            return self.delete_last()
        current = self.head
        for _ in range(index - 1):
            current = current.next
        data = current.next.data
        current.next = current.next.next
        self.size -= 1
        return data

    def search(self, data: Any) -> Optional[Node]:
        """Search for a node with matching data. O(n)"""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        values = []
        current = self.head
        while current:
            values.append(str(current.data))
            current = current.next
        return " -> ".join(values) if values else "Empty"


class LinkedListHeadTail:
    """Singly linked list with head and tail references."""

    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.size: int = 0

    def is_empty(self) -> bool:
        return self.head is None

    def prepend(self, data: Any) -> None:
        """Add node at the beginning. O(1)"""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self.size += 1

    def append(self, data: Any) -> None:
        """Add node at the end. O(1) - tail reference makes this efficient"""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def insert(self, data: Any, index: int) -> bool:
        """Insert node at specific index. O(n)"""
        if index < 0 or index > self.size:
            return False
        if index == 0:
            self.prepend(data)
            return True
        if index == self.size:
            self.append(data)
            return True
        new_node = Node(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        return True

    def delete(self, data: Any) -> bool:
        """Delete first node with matching data. O(n)"""
        if self.head is None:
            return False

        if self.head.data == data:
            if self.head == self.tail:
                self.head = None
                self.tail = None
            else:
                self.head = self.head.next
            self.size -= 1
            return True

        current = self.head
        while current.next:
            if current.next.data == data:
                if current.next == self.tail:
                    self.tail = current
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        return False

    def delete_first(self) -> Optional[Any]:
        """Delete node from the beginning. O(1)"""
        if self.head is None:
            return None
        data = self.head.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
        self.size -= 1
        return data

    def delete_last(self) -> Optional[Any]:
        """Delete node from the end. O(n) - must traverse to find second-to-last"""
        if self.head is None:
            return None
        data = self.tail.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            current = self.head
            while current.next != self.tail:
                current = current.next
            current.next = None
            self.tail = current
        self.size -= 1
        return data

    def delete_at(self, index: int) -> Optional[Any]:
        """Delete node at specific index. O(n)"""
        if index < 0 or index >= self.size:
            return None
        if index == 0:
            return self.delete_first()
        if index == self.size - 1:
            return self.delete_last()
        current = self.head
        for _ in range(index - 1):
            current = current.next
        data = current.next.data
        current.next = current.next.next
        self.size -= 1
        return data

    def search(self, data: Any) -> Optional[Node]:
        """Search for a node with matching data. O(n)"""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        values = []
        current = self.head
        while current:
            values.append(str(current.data))
            current = current.next
        return " -> ".join(values) if values else "Empty"


class DoublyNode:
    """A node in a doubly linked list."""

    def __init__(self, data: Any) -> None:
        self.data: Any = data
        self.prev: Optional[DoublyNode] = None
        self.next: Optional[DoublyNode] = None


class DoublyLinkedList:
    """Doubly linked list with head and tail references."""

    def __init__(self) -> None:
        self.head: Optional[DoublyNode] = None
        self.tail: Optional[DoublyNode] = None
        self.size: int = 0

    def is_empty(self) -> bool:
        return self.head is None

    def prepend(self, data: Any) -> None:
        """Add node at the beginning. O(1)"""
        new_node = DoublyNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1

    def append(self, data: Any) -> None:
        """Add node at the end. O(1)"""
        new_node = DoublyNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def insert(self, data: Any, index: int) -> bool:
        """Insert node at specific index. O(n)"""
        if index < 0 or index > self.size:
            return False
        if index == 0:
            self.prepend(data)
            return True
        if index == self.size:
            self.append(data)
            return True
        new_node = DoublyNode(data)
        current = self.head
        for _ in range(index):
            current = current.next
        new_node.prev = current.prev
        new_node.next = current
        current.prev.next = new_node
        current.prev = new_node
        self.size += 1
        return True

    def delete(self, data: Any) -> bool:
        """Delete first node with matching data. O(n)"""
        current = self.head
        while current:
            if current.data == data:
                # Update previous node's next pointer
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next

                # Update next node's prev pointer
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev

                self.size -= 1
                return True
            current = current.next
        return False

    def delete_first(self) -> Optional[Any]:
        """Delete node from the beginning. O(1)"""
        if self.head is None:
            return None
        data = self.head.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        self.size -= 1
        return data

    def delete_last(self) -> Optional[Any]:
        """Delete node from the end. O(1) - advantage of doubly linked"""
        if self.tail is None:
            return None
        data = self.tail.data
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        self.size -= 1
        return data

    def delete_at(self, index: int) -> Optional[Any]:
        """Delete node at specific index. O(n)"""
        if index < 0 or index >= self.size:
            return None
        if index == 0:
            return self.delete_first()
        if index == self.size - 1:
            return self.delete_last()
        current = self.head
        for _ in range(index):
            current = current.next
        data = current.data
        current.prev.next = current.next
        current.next.prev = current.prev
        self.size -= 1
        return data

    def search(self, data: Any) -> Optional[DoublyNode]:
        """Search for a node with matching data. O(n)"""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None

    def reverse_traverse(self) -> list[Any]:
        """Traverse list from tail to head. O(n)"""
        values: list[Any] = []
        current = self.tail
        while current:
            values.append(current.data)
            current = current.prev
        return values

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        values = []
        current = self.head
        while current:
            values.append(str(current.data))
            current = current.next
        return " <-> ".join(values) if values else "Empty"


def test_linked_list_head_only():
    print("=" * 50)
    print("Testing LinkedListHeadOnly")
    print("=" * 50)

    ll = LinkedListHeadOnly()

    # Test empty list
    print(f"\n1. Empty list: {ll}")
    print(f"   is_empty(): {ll.is_empty()}")
    print(f"   len(): {len(ll)}")

    # Test prepend
    print("\n2. Testing prepend():")
    ll.prepend(3)
    ll.prepend(2)
    ll.prepend(1)
    print(f"   After prepending 3, 2, 1: {ll}")

    # Test append
    print("\n3. Testing append():")
    ll.append(4)
    ll.append(5)
    print(f"   After appending 4, 5: {ll}")
    print(f"   Length: {len(ll)}")

    # Test insert
    print("\n4. Testing insert():")
    print(f"   Before: {ll}")
    ll.insert(99, 0)  # Insert at head
    print(f"   After insert(99, 0) [head]: {ll}")
    ll.insert(88, len(ll))  # Insert at tail
    print(f"   After insert(88, {len(ll)-1}) [tail]: {ll}")
    ll.insert(77, 3)  # Insert at middle
    print(f"   After insert(77, 3) [middle]: {ll}")
    result = ll.insert(66, 100)  # Invalid index
    print(f"   insert(66, 100) returned: {result}")

    # Test search
    print("\n5. Testing search():")
    node = ll.search(3)
    print(f"   search(3): Found node with data = {node.data if node else 'None'}")
    node = ll.search(999)
    print(f"   search(999): {node}")

    # Test delete_first
    print("\n6. Testing delete_first():")
    print(f"   Before: {ll}")
    removed = ll.delete_first()
    print(f"   Removed: {removed}, After: {ll}")

    # Test delete_last
    print("\n7. Testing delete_last():")
    print(f"   Before: {ll}")
    removed = ll.delete_last()
    print(f"   Removed: {removed}, After: {ll}")

    # Test delete_at
    print("\n8. Testing delete_at():")
    print(f"   Before: {ll}")
    removed = ll.delete_at(2)  # Delete middle
    print(f"   delete_at(2) removed: {removed}, After: {ll}")
    removed = ll.delete_at(0)  # Delete head
    print(f"   delete_at(0) removed: {removed}, After: {ll}")
    removed = ll.delete_at(100)  # Invalid index
    print(f"   delete_at(100) returned: {removed}")

    # Test delete by value
    print("\n9. Testing delete():")
    print(f"   Before: {ll}")
    ll.delete(3)
    print(f"   After delete(3): {ll}")
    result = ll.delete(999)
    print(f"   delete(999) returned: {result}")

    # Test edge case: delete until empty
    print("\n10. Testing delete until empty:")
    while not ll.is_empty():
        ll.delete_first()
    print(f"   After deleting all: {ll}")
    print(f"   is_empty(): {ll.is_empty()}")
    print(f"   delete_first() on empty: {ll.delete_first()}")
    print(f"   delete_last() on empty: {ll.delete_last()}")


def test_linked_list_head_tail():
    print("\n" + "=" * 50)
    print("Testing LinkedListHeadTail")
    print("=" * 50)

    ll = LinkedListHeadTail()

    # Test empty list
    print(f"\n1. Empty list: {ll}")
    print(f"   is_empty(): {ll.is_empty()}")
    print(f"   head: {ll.head}, tail: {ll.tail}")

    # Test prepend
    print("\n2. Testing prepend():")
    ll.prepend(3)
    print(f"   After prepending 3: {ll}")
    print(f"   head.data: {ll.head.data}, tail.data: {ll.tail.data}")
    ll.prepend(2)
    ll.prepend(1)
    print(f"   After prepending 2, 1: {ll}")
    print(f"   head.data: {ll.head.data}, tail.data: {ll.tail.data}")

    # Test append
    print("\n3. Testing append():")
    ll.append(4)
    ll.append(5)
    print(f"   After appending 4, 5: {ll}")
    print(f"   head.data: {ll.head.data}, tail.data: {ll.tail.data}")

    # Test insert
    print("\n4. Testing insert():")
    print(f"   Before: {ll}")
    ll.insert(99, 0)  # Insert at head
    print(f"   After insert(99, 0) [head]: {ll}")
    print(f"   head.data: {ll.head.data}, tail.data: {ll.tail.data}")
    ll.insert(88, len(ll))  # Insert at tail
    print(f"   After insert(88, {len(ll)-1}) [tail]: {ll}")
    print(f"   head.data: {ll.head.data}, tail.data: {ll.tail.data}")
    ll.insert(77, 3)  # Insert at middle
    print(f"   After insert(77, 3) [middle]: {ll}")

    # Test search
    print("\n5. Testing search():")
    node = ll.search(3)
    print(f"   search(3): Found node with data = {node.data if node else 'None'}")
    node = ll.search(999)
    print(f"   search(999): {node}")

    # Test delete_first
    print("\n6. Testing delete_first():")
    print(f"   Before: {ll}")
    removed = ll.delete_first()
    print(f"   Removed: {removed}, After: {ll}")
    print(f"   head.data: {ll.head.data}, tail.data: {ll.tail.data}")

    # Test delete_last
    print("\n7. Testing delete_last():")
    print(f"   Before: {ll}")
    removed = ll.delete_last()
    print(f"   Removed: {removed}, After: {ll}")
    print(f"   head.data: {ll.head.data}, tail.data: {ll.tail.data}")

    # Test delete_at
    print("\n8. Testing delete_at():")
    print(f"   Before: {ll}")
    removed = ll.delete_at(2)  # Delete middle
    print(f"   delete_at(2) removed: {removed}, After: {ll}")
    removed = ll.delete_at(0)  # Delete head
    print(f"   delete_at(0) removed: {removed}, After: {ll}")
    print(f"   head.data: {ll.head.data}, tail.data: {ll.tail.data}")

    # Test delete by value
    print("\n9. Testing delete():")
    print(f"   Before: {ll}")
    ll.delete(3)
    print(f"   After delete(3): {ll}")

    # Test edge case: delete until empty
    print("\n10. Testing delete until empty:")
    while not ll.is_empty():
        ll.delete_first()
    print(f"   After deleting all: {ll}")
    print(f"   is_empty(): {ll.is_empty()}")
    print(f"   head: {ll.head}, tail: {ll.tail}")
    print(f"   delete_first() on empty: {ll.delete_first()}")
    print(f"   delete_last() on empty: {ll.delete_last()}")


def test_doubly_linked_list():
    print("\n" + "=" * 50)
    print("Testing DoublyLinkedList")
    print("=" * 50)

    dll = DoublyLinkedList()

    # Test empty list
    print(f"\n1. Empty list: {dll}")
    print(f"   is_empty(): {dll.is_empty()}")
    print(f"   head: {dll.head}, tail: {dll.tail}")

    # Test prepend
    print("\n2. Testing prepend():")
    dll.prepend(3)
    print(f"   After prepending 3: {dll}")
    print(f"   head.data: {dll.head.data}, tail.data: {dll.tail.data}")
    dll.prepend(2)
    dll.prepend(1)
    print(f"   After prepending 2, 1: {dll}")
    print(f"   head.data: {dll.head.data}, tail.data: {dll.tail.data}")

    # Test append
    print("\n3. Testing append():")
    dll.append(4)
    dll.append(5)
    print(f"   After appending 4, 5: {dll}")
    print(f"   head.data: {dll.head.data}, tail.data: {dll.tail.data}")

    # Test insert
    print("\n4. Testing insert():")
    print(f"   Before: {dll}")
    dll.insert(99, 0)  # Insert at head
    print(f"   After insert(99, 0) [head]: {dll}")
    print(f"   head.data: {dll.head.data}, head.prev: {dll.head.prev}")
    dll.insert(88, len(dll))  # Insert at tail
    print(f"   After insert(88, {len(dll)-1}) [tail]: {dll}")
    print(f"   tail.data: {dll.tail.data}, tail.next: {dll.tail.next}")
    dll.insert(77, 3)  # Insert at middle
    print(f"   After insert(77, 3) [middle]: {dll}")

    # Test prev/next pointers
    print("\n5. Testing prev/next pointers:")
    node = dll.search(3)
    print(f"   Node 3's prev: {node.prev.data if node.prev else None}")
    print(f"   Node 3's next: {node.next.data if node.next else None}")

    # Test reverse traverse
    print("\n6. Testing reverse_traverse():")
    print(f"   Forward:  {dll}")
    print(f"   Backward: {dll.reverse_traverse()}")

    # Test search
    print("\n7. Testing search():")
    node = dll.search(3)
    print(f"   search(3): Found node with data = {node.data if node else 'None'}")
    node = dll.search(999)
    print(f"   search(999): {node}")

    # Test delete_first
    print("\n8. Testing delete_first():")
    print(f"   Before: {dll}")
    removed = dll.delete_first()
    print(f"   Removed: {removed}, After: {dll}")
    print(f"   head.data: {dll.head.data}, head.prev: {dll.head.prev}")

    # Test delete_last (O(1) for doubly linked!)
    print("\n9. Testing delete_last() [O(1) advantage]:")
    print(f"   Before: {dll}")
    removed = dll.delete_last()
    print(f"   Removed: {removed}, After: {dll}")
    print(f"   tail.data: {dll.tail.data}, tail.next: {dll.tail.next}")

    # Test delete_at
    print("\n10. Testing delete_at():")
    print(f"   Before: {dll}")
    removed = dll.delete_at(2)  # Delete middle
    print(f"   delete_at(2) removed: {removed}, After: {dll}")
    removed = dll.delete_at(0)  # Delete head
    print(f"   delete_at(0) removed: {removed}, After: {dll}")
    print(f"   head.data: {dll.head.data}, tail.data: {dll.tail.data}")

    # Test delete by value
    print("\n11. Testing delete():")
    print(f"   Before: {dll}")
    dll.delete(3)
    print(f"   After delete(3): {dll}")

    # Test edge case: delete until empty
    print("\n12. Testing delete until empty:")
    while not dll.is_empty():
        dll.delete_first()
    print(f"   After deleting all: {dll}")
    print(f"   is_empty(): {dll.is_empty()}")
    print(f"   head: {dll.head}, tail: {dll.tail}")
    print(f"   delete_first() on empty: {dll.delete_first()}")
    print(f"   delete_last() on empty: {dll.delete_last()}")


def test_comparison():
    print("\n" + "=" * 50)
    print("Comparison: All Linked List Types")
    print("=" * 50)

    # Time complexity comparison table
    print("\n1. Time Complexity Comparison:")
    print("   +------------------+-------------+-------------+---------------+")
    print("   | Operation        | Head Only   | Head+Tail   | Doubly Linked |")
    print("   +------------------+-------------+-------------+---------------+")
    print("   | prepend()        | O(1)        | O(1)        | O(1)          |")
    print("   | append()         | O(n)        | O(1)        | O(1)          |")
    print("   | delete(value)    | O(n)        | O(n)        | O(n)          |")
    print("   | delete_last      | O(n)        | O(n)        | O(1)          |")
    print("   | search()         | O(n)        | O(n)        | O(n)          |")
    print("   | reverse_traverse | O(n) rebuild| O(n) rebuild| O(n) direct   |")
    print("   +------------------+-------------+-------------+---------------+")

    # Test append
    print("\n2. Testing append():")
    ll1 = LinkedListHeadOnly()
    ll2 = LinkedListHeadTail()
    dll = DoublyLinkedList()

    for i in range(5):
        ll1.append(i)
        ll2.append(i)
        dll.append(i)

    print(f"   Head Only:      {ll1}")
    print(f"   Head+Tail:      {ll2}")
    print(f"   Doubly Linked:  {dll}")

    # Test prepend
    print("\n3. Testing prepend():")
    ll1 = LinkedListHeadOnly()
    ll2 = LinkedListHeadTail()
    dll = DoublyLinkedList()

    for i in range(5):
        ll1.prepend(i)
        ll2.prepend(i)
        dll.prepend(i)

    print(f"   Head Only:      {ll1}")
    print(f"   Head+Tail:      {ll2}")
    print(f"   Doubly Linked:  {dll}")

    # Test delete from middle
    print("\n4. Testing delete() from middle:")
    ll1 = LinkedListHeadOnly()
    ll2 = LinkedListHeadTail()
    dll = DoublyLinkedList()

    for i in range(5):
        ll1.append(i)
        ll2.append(i)
        dll.append(i)

    print(f"   Before delete(2):")
    print(f"   Head Only:      {ll1}")
    print(f"   Head+Tail:      {ll2}")
    print(f"   Doubly Linked:  {dll}")

    ll1.delete(2)
    ll2.delete(2)
    dll.delete(2)

    print(f"   After delete(2):")
    print(f"   Head Only:      {ll1}")
    print(f"   Head+Tail:      {ll2}")
    print(f"   Doubly Linked:  {dll}")

    # Test delete from tail - advantage of doubly linked
    print("\n5. Testing delete from tail (Doubly Linked advantage):")
    ll1 = LinkedListHeadOnly()
    ll2 = LinkedListHeadTail()
    dll = DoublyLinkedList()

    for i in range(5):
        ll1.append(i)
        ll2.append(i)
        dll.append(i)

    print(f"   Before:")
    print(f"   Head Only:      {ll1}")
    print(f"   Head+Tail:      {ll2}")
    print(f"   Doubly Linked:  {dll}")

    # For singly linked, must traverse to find second-to-last
    ll1.delete(4)
    ll2.delete(4)
    removed = dll.delete_last()  # O(1) for doubly linked!

    print(f"   After removing last element (removed {removed} from DLL):")
    print(f"   Head Only:      {ll1}  (O(n) - must traverse)")
    print(f"   Head+Tail:      {ll2}  (O(n) - must traverse)")
    print(f"   Doubly Linked:  {dll}  (O(1) - direct tail access)")

    # Test reverse traverse - advantage of doubly linked
    print("\n6. Testing reverse traverse (Doubly Linked advantage):")
    dll = DoublyLinkedList()
    for i in range(5):
        dll.append(i)

    print(f"   Forward:  {dll}")
    print(f"   Backward: {dll.reverse_traverse()}")
    print("   (Singly linked lists would need O(n) space to reverse)")

    # Memory comparison
    print("\n7. Memory Usage Comparison:")
    print("   Head Only:      Node(data, next)         - 2 fields per node")
    print("   Head+Tail:      Node(data, next)         - 2 fields per node + tail ref")
    print("   Doubly Linked:  Node(data, prev, next)   - 3 fields per node")

    # Use case recommendations
    print("\n8. Use Case Recommendations:")
    print("   Head Only:      Simple stack, memory constrained")
    print("   Head+Tail:      Queue, frequent append operations")
    print("   Doubly Linked:  Deque, LRU cache, browser history")
