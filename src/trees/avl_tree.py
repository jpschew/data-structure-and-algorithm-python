from typing import Any, Optional


class AVLNode:
    """A node in an AVL tree with height attribute."""

    def __init__(self, data: Any) -> None:
        self.data: Any = data
        self.left: Optional[AVLNode] = None
        self.right: Optional[AVLNode] = None
        self.height: int = 0  # Height of node (leaf = 0)


class AVLTree:
    """
    AVL Tree implementation - a self-balancing Binary Search Tree.

    Balance Property: For every node, |height(left) - height(right)| <= 1

    Time Complexity (guaranteed):
    - insert(): O(log n)
    - delete(): O(log n)
    - search(): O(log n)

    Unlike regular BST, AVL never degenerates to O(n) due to automatic balancing.
    """

    def __init__(self) -> None:
        self.root: Optional[AVLNode] = None
        self.size: int = 0

    def is_empty(self) -> bool:
        """Check if tree is empty. O(1)"""
        return self.root is None

    # =========================================================================
    # Height and Balance Factor
    # =========================================================================

    def _get_height(self, node: Optional[AVLNode]) -> int:
        """Get height of a node. Returns -1 for None."""
        if node is None:
            return -1
        return node.height

    def _update_height(self, node: AVLNode) -> None:
        """Update height of a node based on children heights."""
        node.height = max(self._get_height(node.left), self._get_height(node.right)) + 1

    def _get_balance_factor(self, node: Optional[AVLNode]) -> int:
        """
        Get balance factor of a node.
        Balance factor = height(left) - height(right)

        - Positive: Left-heavy
        - Negative: Right-heavy
        - |balance| > 1: Unbalanced, needs rotation
        """
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    # =========================================================================
    # Rotations
    # =========================================================================

    def _rotate_right(self, y: AVLNode) -> AVLNode:
        """
        Right rotation (for Left-Left case).

            y                x
           / \              / \
          x   C    -->     A   y
         / \                  / \
        A   B                B   C

        Returns new root of subtree (x).
        """
        x = y.left
        B = x.right

        # Perform rotation
        x.right = y
        y.left = B

        # Update heights (y first, then x)
        self._update_height(y)
        self._update_height(x)

        return x

    def _rotate_left(self, x: AVLNode) -> AVLNode:
        """
        Left rotation (for Right-Right case).

          x                  y
         / \                / \
        A   y     -->      x   C
           / \            / \
          B   C          A   B

        Returns new root of subtree (y).
        """
        y = x.right
        B = y.left

        # Perform rotation
        y.left = x
        x.right = B

        # Update heights (x first, then y)
        self._update_height(x)
        self._update_height(y)

        return y

    # =========================================================================
    # Rebalancing
    # =========================================================================

    def _rebalance(self, node: AVLNode) -> AVLNode:
        """
        Rebalance a node if needed. Returns the new root of the subtree.

        Four cases:
        1. Left-Left (LL):   balance > 1 and left child is left-heavy or balanced
        2. Left-Right (LR):  balance > 1 and left child is right-heavy
        3. Right-Right (RR): balance < -1 and right child is right-heavy or balanced
        4. Right-Left (RL):  balance < -1 and right child is left-heavy
        """
        self._update_height(node)
        balance = self._get_balance_factor(node)

        # Left-heavy (balance > 1)
        if balance > 1:
            # Left-Right case: first rotate left child left, then rotate node right
            if self._get_balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            # Left-Left case (or after LR adjustment): rotate right
            return self._rotate_right(node)

        # Right-heavy (balance < -1)
        if balance < -1:
            # Right-Left case: first rotate right child right, then rotate node left
            if self._get_balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            # Right-Right case (or after RL adjustment): rotate left
            return self._rotate_left(node)

        return node

    # =========================================================================
    # Insert
    # =========================================================================

    def insert(self, data: Any) -> None:
        """Insert a value into the AVL tree. O(log n)"""
        self.root = self._insert_recursive(self.root, data)
        self.size += 1

    def _insert_recursive(self, node: Optional[AVLNode], data: Any) -> AVLNode:
        # Standard BST insert
        if node is None:
            return AVLNode(data)

        if data < node.data:
            node.left = self._insert_recursive(node.left, data)
        else:
            node.right = self._insert_recursive(node.right, data)

        # Rebalance and return
        return self._rebalance(node)

    # =========================================================================
    # Delete
    # =========================================================================

    def delete(self, data: Any) -> bool:
        """Delete a value from the AVL tree. O(log n)"""
        self.root, deleted = self._delete_recursive(self.root, data)
        if deleted:
            self.size -= 1
        return deleted

    def _delete_recursive(
        self, node: Optional[AVLNode], data: Any
    ) -> tuple[Optional[AVLNode], bool]:
        if node is None:
            return None, False

        deleted = False

        if data < node.data:
            node.left, deleted = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right, deleted = self._delete_recursive(node.right, data)
        else:
            # Node found
            deleted = True

            # Case 1: No children or one child
            if node.left is None:
                return node.right, deleted
            if node.right is None:
                return node.left, deleted

            # Case 2: Two children - find inorder successor
            successor = self._find_min_node(node.right)
            node.data = successor.data
            node.right, _ = self._delete_recursive(node.right, successor.data)

        # Rebalance and return
        return self._rebalance(node), deleted

    # =========================================================================
    # Search
    # =========================================================================

    def search(self, data: Any) -> Optional[AVLNode]:
        """Search for a value in the AVL tree. O(log n)"""
        return self._search_recursive(self.root, data)

    def _search_recursive(self, node: Optional[AVLNode], data: Any) -> Optional[AVLNode]:
        if node is None:
            return None
        if data == node.data:
            return node
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)

    # =========================================================================
    # Min / Max
    # =========================================================================

    def _find_min_node(self, node: AVLNode) -> AVLNode:
        """Find the minimum value node in a subtree."""
        current = node
        while current.left:
            current = current.left
        return current

    def find_min(self) -> Optional[Any]:
        """Find minimum value in AVL tree. O(log n)"""
        if self.root is None:
            return None
        return self._find_min_node(self.root).data

    def _find_max_node(self, node: AVLNode) -> AVLNode:
        """Find the maximum value node in a subtree."""
        current = node
        while current.right:
            current = current.right
        return current

    def find_max(self) -> Optional[Any]:
        """Find maximum value in AVL tree. O(log n)"""
        if self.root is None:
            return None
        return self._find_max_node(self.root).data

    # =========================================================================
    # Traversals
    # =========================================================================

    def inorder(self) -> list[Any]:
        """Inorder traversal (Left, Root, Right) - returns sorted order. O(n)"""
        result: list[Any] = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node: Optional[AVLNode], result: list[Any]) -> None:
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)

    def preorder(self) -> list[Any]:
        """Preorder traversal (Root, Left, Right). O(n)"""
        result: list[Any] = []
        self._preorder_recursive(self.root, result)
        return result

    def _preorder_recursive(self, node: Optional[AVLNode], result: list[Any]) -> None:
        if node:
            result.append(node.data)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)

    def level_order(self) -> list[Any]:
        """Level order traversal (BFS). O(n)"""
        if self.root is None:
            return []
        result: list[Any] = []
        queue: list[AVLNode] = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node.data)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result

    # =========================================================================
    # Utility
    # =========================================================================

    def height(self) -> int:
        """Get height of the tree. O(1)"""
        return self._get_height(self.root)

    def is_balanced(self) -> bool:
        """Check if tree is balanced (should always be True for AVL). O(n)"""
        return self._is_balanced_recursive(self.root)

    def _is_balanced_recursive(self, node: Optional[AVLNode]) -> bool:
        if node is None:
            return True
        balance = self._get_balance_factor(node)
        if abs(balance) > 1:
            return False
        return self._is_balanced_recursive(node.left) and self._is_balanced_recursive(
            node.right
        )

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        if self.root is None:
            return "Empty"
        return f"AVL(size={self.size}, height={self.height()}, root={self.root.data}, inorder={self.inorder()})"


def test_avl_tree():
    print("=" * 60)
    print("Testing AVLTree")
    print("=" * 60)

    avl = AVLTree()

    # Test empty tree
    print(f"\n1. Empty tree: {avl}")
    print(f"   is_empty(): {avl.is_empty()}")
    print(f"   height(): {avl.height()}")

    # Test insert (will trigger rotations)
    print("\n2. Testing insert() with rotations:")

    # Insert in ascending order (would be degenerate in BST)
    print("   Inserting: 1, 2, 3, 4, 5, 6, 7 (ascending order)")
    for val in [1, 2, 3, 4, 5, 6, 7]:
        avl.insert(val)
        print(f"   After insert({val}): height={avl.height()}, balanced={avl.is_balanced()}")

    print(f"\n   Final tree: {avl}")
    print(f"   Inorder: {avl.inorder()}")
    print(f"   Preorder: {avl.preorder()}")
    print(f"   Level order: {avl.level_order()}")

    # Compare with BST (would have height 6)
    print(f"\n   BST height for same data: 6 (degenerate)")
    print(f"   AVL height: {avl.height()} (balanced)")

    # Test search
    print("\n3. Testing search():")
    node = avl.search(4)
    print(f"   search(4): Found node with data = {node.data if node else 'None'}")
    node = avl.search(99)
    print(f"   search(99): {node}")

    # Test min/max
    print("\n4. Testing find_min() and find_max():")
    print(f"   find_min(): {avl.find_min()}")
    print(f"   find_max(): {avl.find_max()}")

    # Test delete
    print("\n5. Testing delete():")
    print(f"   Before: {avl.inorder()}, height={avl.height()}")

    avl.delete(4)  # Delete root
    print(f"   After delete(4): {avl.inorder()}, height={avl.height()}, balanced={avl.is_balanced()}")

    avl.delete(1)
    print(f"   After delete(1): {avl.inorder()}, height={avl.height()}, balanced={avl.is_balanced()}")

    avl.delete(7)
    print(f"   After delete(7): {avl.inorder()}, height={avl.height()}, balanced={avl.is_balanced()}")

    # Test rotation cases
    print("\n6. Testing specific rotation cases:")

    # Left-Left case
    print("\n   Left-Left (LL) case:")
    ll_tree = AVLTree()
    print("   Inserting: 30, 20, 10")
    for val in [30, 20, 10]:
        ll_tree.insert(val)
    print(f"   Result: preorder={ll_tree.preorder()}, balanced={ll_tree.is_balanced()}")
    print("   Expected: [20, 10, 30] (right rotation)")

    # Right-Right case
    print("\n   Right-Right (RR) case:")
    rr_tree = AVLTree()
    print("   Inserting: 10, 20, 30")
    for val in [10, 20, 30]:
        rr_tree.insert(val)
    print(f"   Result: preorder={rr_tree.preorder()}, balanced={rr_tree.is_balanced()}")
    print("   Expected: [20, 10, 30] (left rotation)")

    # Left-Right case
    print("\n   Left-Right (LR) case:")
    lr_tree = AVLTree()
    print("   Inserting: 30, 10, 20")
    for val in [30, 10, 20]:
        lr_tree.insert(val)
    print(f"   Result: preorder={lr_tree.preorder()}, balanced={lr_tree.is_balanced()}")
    print("   Expected: [20, 10, 30] (left-right rotation)")

    # Right-Left case
    print("\n   Right-Left (RL) case:")
    rl_tree = AVLTree()
    print("   Inserting: 10, 30, 20")
    for val in [10, 30, 20]:
        rl_tree.insert(val)
    print(f"   Result: preorder={rl_tree.preorder()}, balanced={rl_tree.is_balanced()}")
    print("   Expected: [20, 10, 30] (right-left rotation)")
