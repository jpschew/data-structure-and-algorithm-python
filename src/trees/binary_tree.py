class TreeNode:
    """A node in a binary tree."""

    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


class BinarySearchTree:
    """Binary Search Tree implementation (left < root < right)."""

    def __init__(self):
        self.root = None
        self.size = 0

    def is_empty(self):
        """Check if tree is empty. O(1)"""
        return self.root is None

    def insert(self, data):
        """Insert a value into the BST. O(log n) average, O(n) worst"""
        if self.root is None:
            self.root = TreeNode(data)
        else:
            self._insert_recursive(self.root, data)
        self.size += 1

    def _insert_recursive(self, node, data):
        if data < node.data:
            if node.left is None:
                node.left = TreeNode(data)
            else:
                self._insert_recursive(node.left, data)
        else:
            if node.right is None:
                node.right = TreeNode(data)
            else:
                self._insert_recursive(node.right, data)

    def search(self, data):
        """Search for a value in the BST. O(log n) average, O(n) worst"""
        return self._search_recursive(self.root, data)

    def _search_recursive(self, node, data):
        if node is None:
            return None
        if data == node.data:
            return node
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)

    def delete(self, data):
        """Delete a value from the BST. O(log n) average, O(n) worst"""
        self.root, deleted = self._delete_recursive(self.root, data)
        if deleted:
            self.size -= 1
        return deleted

    def _delete_recursive(self, node, data):
        if node is None:
            return node, False

        deleted = False
        if data < node.data:
            node.left, deleted = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right, deleted = self._delete_recursive(node.right, data)
        else:
            # Node found
            deleted = True
            # Case 1: No children
            if node.left is None and node.right is None:
                return None, deleted
            # Case 2: One child
            if node.left is None:
                return node.right, deleted
            if node.right is None:
                return node.left, deleted
            # Case 3: Two children - find inorder successor
            successor = self._find_min(node.right)
            node.data = successor.data
            node.right, _ = self._delete_recursive(node.right, successor.data)

        return node, deleted

    def _find_min(self, node):
        """Find the minimum value node in a subtree."""
        current = node
        while current.left:
            current = current.left
        return current

    def find_min(self):
        """Find minimum value in BST. O(log n) average, O(n) worst"""
        if self.root is None:
            return None
        return self._find_min(self.root).data

    def find_max(self):
        """Find maximum value in BST. O(log n) average, O(n) worst"""
        if self.root is None:
            return None
        current = self.root
        while current.right:
            current = current.right
        return current.data

    # Tree Traversals
    def inorder(self):
        """Inorder traversal (Left, Root, Right) - returns sorted order. O(n)"""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)

    def preorder(self):
        """Preorder traversal (Root, Left, Right). O(n)"""
        result = []
        self._preorder_recursive(self.root, result)
        return result

    def _preorder_recursive(self, node, result):
        if node:
            result.append(node.data)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)

    def postorder(self):
        """Postorder traversal (Left, Right, Root). O(n)"""
        result = []
        self._postorder_recursive(self.root, result)
        return result

    def _postorder_recursive(self, node, result):
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.data)

    def level_order(self):
        """Level order traversal (BFS). O(n)"""
        if self.root is None:
            return []
        result = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node.data)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result

    def height(self):
        """Get height of the tree. O(n)"""
        return self._height_recursive(self.root)

    def _height_recursive(self, node):
        if node is None:
            return -1
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return max(left_height, right_height) + 1

    def __len__(self):
        return self.size

    def __str__(self):
        if self.root is None:
            return "Empty"
        return f"BST(size={self.size}, root={self.root.data}, inorder={self.inorder()})"


def test_binary_search_tree():
    print("=" * 50)
    print("Testing BinarySearchTree")
    print("=" * 50)

    bst = BinarySearchTree()

    # Test empty tree
    print(f"\n1. Empty tree: {bst}")
    print(f"   is_empty(): {bst.is_empty()}")
    print(f"   height(): {bst.height()}")

    # Test insert
    print("\n2. Testing insert():")
    #        50
    #       /  \
    #      30   70
    #     / \   / \
    #    20 40 60 80
    for val in [50, 30, 70, 20, 40, 60, 80]:
        bst.insert(val)
    print(f"   Inserted: 50, 30, 70, 20, 40, 60, 80")
    print(f"   Size: {len(bst)}")
    print(f"   Height: {bst.height()}")

    # Test traversals
    print("\n3. Testing traversals:")
    print(f"   Inorder (sorted):  {bst.inorder()}")
    print(f"   Preorder:          {bst.preorder()}")
    print(f"   Postorder:         {bst.postorder()}")
    print(f"   Level order (BFS): {bst.level_order()}")

    # Test search
    print("\n4. Testing search():")
    node = bst.search(40)
    print(f"   search(40): Found node with data = {node.data if node else 'None'}")
    node = bst.search(99)
    print(f"   search(99): {node}")

    # Test min/max
    print("\n5. Testing find_min() and find_max():")
    print(f"   find_min(): {bst.find_min()}")
    print(f"   find_max(): {bst.find_max()}")

    # Test delete
    print("\n6. Testing delete():")
    print(f"   Before: {bst.inorder()}")

    bst.delete(20)  # Delete leaf
    print(f"   After delete(20) [leaf]: {bst.inorder()}")

    bst.delete(30)  # Delete node with one child
    print(f"   After delete(30) [one child]: {bst.inorder()}")

    bst.delete(50)  # Delete node with two children (root)
    print(f"   After delete(50) [two children/root]: {bst.inorder()}")

    result = bst.delete(99)  # Delete non-existent
    print(f"   delete(99) returned: {result}")

    print(f"\n   Final size: {len(bst)}")
    print(f"   Final height: {bst.height()}")
