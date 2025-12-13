from __future__ import annotations

from typing import Optional


class TrieNode:
    """A node in a Trie (prefix tree)."""

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Trie:
    """
    Trie (Prefix Tree) implementation.

    A trie is a tree-like data structure used for efficient retrieval of keys
    in a dataset of strings. Common use cases include autocomplete, spell
    checking, and IP routing.

    Time Complexity:
        - insert: O(m) where m is the length of the word
        - search: O(m)
        - starts_with: O(m)
        - delete: O(m)

    Space Complexity: O(n * m) where n is number of words, m is average length
    """

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()
        self.word_count: int = 0

    def insert(self, word: str) -> None:
        """Insert a word into the trie. O(m) where m is word length."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1

    def search(self, word: str) -> bool:
        """Search for an exact word in the trie. O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Check if any word in the trie starts with the given prefix. O(m)"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find the node corresponding to the last character of prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie. O(m)
        Returns True if word was deleted, False if word not found.
        """
        if not word:
            return False
        self._delete_recursive(self.root, word, 0)
        return True

    def _has_children(self, node: TrieNode) -> bool:
        """Check if node has any children."""
        return len(node.children) > 0

    def _delete_recursive(self, node: TrieNode, word: str, index: int) -> bool:
        """Helper for delete operation."""
        if index == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            self.word_count -= 1
            return not self._has_children(node)

        char = word[index]
        if char not in node.children:
            return False

        should_delete_child = self._delete_recursive(
            node.children[char], word, index + 1
        )

        if should_delete_child:
            del node.children[char]
            return not self._has_children(node) and not node.is_end_of_word

        return False

    def get_all_words(self) -> list[str]:
        """Get all words stored in the trie. O(n * m)"""
        words: list[str] = []
        self._collect_words(self.root, "", words)
        return words

    def _collect_words(
        self, node: TrieNode, prefix: str, words: list[str]
    ) -> None:
        """Helper to collect all words from a given node."""
        if node.is_end_of_word:
            words.append(prefix)
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, words)

    def auto_complete(self, prefix: str) -> list[str]:
        """
        Get all words that start with the given prefix (autocomplete). O(n * m)
        Returns empty list if prefix not found.
        """
        node = self._find_node(prefix)
        if node is None:
            return []
        words: list[str] = []
        self._collect_words(node, prefix, words)
        return words

    def count_words_with_prefix(self, prefix: str) -> int:
        """Count how many words start with the given prefix. O(n * m)"""
        return len(self.auto_complete(prefix))

    def is_empty(self) -> bool:
        """Check if the trie is empty. O(1)"""
        return self.word_count == 0

    def __len__(self) -> int:
        """Return the number of words in the trie. O(1)"""
        return self.word_count


class TrieNodeLCRS:
    """
    A node in a Left-Child Right-Sibling Trie.

    Instead of using a dictionary or array to store children,
    each node stores only:
    - char: the character this node represents
    - child: pointer to the first child
    - sibling: pointer to the next sibling

    Structure visualization for "app", "apple", "bat":

        root
         |
         a -----> b
         |        |
         p        a
         |        |
         p*       t*
         |
         l
         |
         e*

    * = is_end_of_word
    | = child pointer
    -> = sibling pointer
    """

    def __init__(self, char: str = "") -> None:
        self.char: str = char
        self.child: Optional[TrieNodeLCRS] = None
        self.sibling: Optional[TrieNodeLCRS] = None
        self.is_end_of_word: bool = False


class TrieLCRS:
    """
    Left-Child Right-Sibling Trie implementation.

    Uses minimal memory - no dict or array per node.
    Each node only has pointers to first child and next sibling.

    Time Complexity:
        - insert: O(m * k) where m is word length, k is avg siblings per level
        - search: O(m * k)
        - starts_with: O(m * k)
        - delete: O(m * k)

    Space Complexity: O(total unique characters) - very memory efficient
    """

    def __init__(self) -> None:
        self.root: TrieNodeLCRS = TrieNodeLCRS()
        self.word_count: int = 0

    def _find_child(self, node: TrieNodeLCRS, char: str) -> Optional[TrieNodeLCRS]:
        """Find a child with the given character by scanning siblings. O(k)"""
        current = node.child
        while current is not None:
            if current.char == char:
                return current
            current = current.sibling
        return None

    def _add_child(self, node: TrieNodeLCRS, char: str) -> TrieNodeLCRS:
        """Add a new child with the given character. O(1)"""
        new_node = TrieNodeLCRS(char)
        # Insert at the beginning of sibling list (O(1))
        new_node.sibling = node.child
        node.child = new_node
        return new_node

    def insert(self, word: str) -> None:
        """Insert a word into the trie. O(m * k)"""
        node = self.root
        for char in word:
            child = self._find_child(node, char)
            if child is None:
                child = self._add_child(node, char)
            node = child
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1

    def search(self, word: str) -> bool:
        """Search for an exact word in the trie. O(m * k)"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Check if any word in the trie starts with the given prefix. O(m * k)"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNodeLCRS]:
        """Find the node corresponding to the last character of prefix."""
        node = self.root
        for char in prefix:
            child = self._find_child(node, char)
            if child is None:
                return None
            node = child
        return node

    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie. O(m * k)
        Returns True if word was deleted, False if word not found.
        """
        if not word:
            return False
        # Find first char node
        first_child = self._find_child(self.root, word[0])
        if first_child is None:
            return False
        self._delete_recursive(self.root, first_child, word, 1)
        return True

    def _has_children(self, node: TrieNodeLCRS) -> bool:
        """Check if node has any children."""
        return node.child is not None

    def _delete_recursive(
        self, parent: TrieNodeLCRS, node: TrieNodeLCRS, word: str, index: int
    ) -> bool:
        """Helper for delete operation."""
        if index == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            self.word_count -= 1
            return not self._has_children(node)

        char = word[index]
        child = self._find_child(node, char)
        if child is None:
            return False

        should_delete_child = self._delete_recursive(node, child, word, index + 1)

        if should_delete_child:
            # Remove child from sibling list
            if node.child == child:
                node.child = child.sibling
            else:
                prev = node.child
                while prev is not None and prev.sibling != child:
                    prev = prev.sibling
                if prev is not None:
                    prev.sibling = child.sibling
            return not self._has_children(node) and not node.is_end_of_word

        return False

    def get_all_words(self) -> list[str]:
        """Get all words stored in the trie. O(n * m)"""
        words: list[str] = []
        self._collect_words(self.root, "", words)
        return words

    def _collect_words(
        self, node: TrieNodeLCRS, prefix: str, words: list[str]
    ) -> None:
        """Helper to collect all words from a given node."""
        # For root node, don't add its char to prefix
        current_prefix = prefix
        if node.char:
            current_prefix = prefix + node.char

        if node.is_end_of_word:
            words.append(current_prefix)

        # Visit all children
        child = node.child
        while child is not None:
            self._collect_words(child, current_prefix, words)
            child = child.sibling

    def auto_complete(self, prefix: str) -> list[str]:
        """
        Get all words that start with the given prefix (autocomplete). O(n * m)
        Returns empty list if prefix not found.
        """
        node = self._find_node(prefix)
        if node is None:
            return []
        words: list[str] = []
        # Collect from the prefix node, but prefix is already built
        if node.is_end_of_word:
            words.append(prefix)
        child = node.child
        while child is not None:
            self._collect_words(child, prefix, words)
            child = child.sibling
        return words

    def count_words_with_prefix(self, prefix: str) -> int:
        """Count how many words start with the given prefix. O(n * m)"""
        return len(self.auto_complete(prefix))

    def is_empty(self) -> bool:
        """Check if the trie is empty. O(1)"""
        return self.word_count == 0

    def __len__(self) -> int:
        """Return the number of words in the trie. O(1)"""
        return self.word_count


class TrieNodeArray:
    """A node in a fixed-array Trie (supports lowercase a-z only)."""

    ALPHABET_SIZE = 26

    def __init__(self) -> None:
        self.children: list[Optional[TrieNodeArray]] = [None] * self.ALPHABET_SIZE
        self.is_end_of_word: bool = False


class TrieArray:
    """
    Fixed-Array Trie implementation (supports lowercase a-z only).

    Each node has a fixed-size array of 26 slots for children.
    Index 0 = 'a', index 1 = 'b', ..., index 25 = 'z'.

    Time Complexity:
        - insert: O(m) where m is the length of the word
        - search: O(m)
        - starts_with: O(m)
        - delete: O(m)

    Space Complexity: O(n * m * 26) - each node has 26 slots
    """

    def __init__(self) -> None:
        self.root: TrieNodeArray = TrieNodeArray()
        self.word_count: int = 0

    def _char_to_index(self, char: str) -> int:
        """Convert character to array index (a=0, b=1, ..., z=25)."""
        return ord(char) - ord("a")

    def _index_to_char(self, index: int) -> str:
        """Convert array index to character."""
        return chr(index + ord("a"))

    def insert(self, word: str) -> None:
        """Insert a word into the trie. O(m)"""
        node = self.root
        for char in word:
            index = self._char_to_index(char)
            if node.children[index] is None:
                node.children[index] = TrieNodeArray()
            node = node.children[index]
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1

    def search(self, word: str) -> bool:
        """Search for an exact word in the trie. O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix. O(m)"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNodeArray]:
        """Find the node corresponding to the last character of prefix."""
        node = self.root
        for char in prefix:
            index = self._char_to_index(char)
            if node.children[index] is None:
                return None
            node = node.children[index]
        return node

    def delete(self, word: str) -> bool:
        """Delete a word from the trie. O(m)"""
        if not word:
            return False
        self._delete_recursive(self.root, word, 0)
        return True

    def _has_children(self, node: TrieNodeArray) -> bool:
        """Check if node has any children."""
        return any(child is not None for child in node.children)

    def _delete_recursive(self, node: TrieNodeArray, word: str, index: int) -> bool:
        """Helper for delete operation."""
        if index == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            self.word_count -= 1
            return not self._has_children(node)

        char_index = self._char_to_index(word[index])
        if node.children[char_index] is None:
            return False

        should_delete_child = self._delete_recursive(
            node.children[char_index], word, index + 1
        )

        if should_delete_child:
            node.children[char_index] = None
            return not self._has_children(node) and not node.is_end_of_word

        return False

    def get_all_words(self) -> list[str]:
        """Get all words stored in the trie. O(n * m)"""
        words: list[str] = []
        self._collect_words(self.root, "", words)
        return words

    def _collect_words(
        self, node: TrieNodeArray, prefix: str, words: list[str]
    ) -> None:
        """Helper to collect all words from a given node."""
        if node.is_end_of_word:
            words.append(prefix)
        for i, child in enumerate(node.children):
            if child is not None:
                self._collect_words(child, prefix + self._index_to_char(i), words)

    def auto_complete(self, prefix: str) -> list[str]:
        """Get all words that start with the given prefix. O(n * m)"""
        node = self._find_node(prefix)
        if node is None:
            return []
        words: list[str] = []
        self._collect_words(node, prefix, words)
        return words

    def count_words_with_prefix(self, prefix: str) -> int:
        """Count how many words start with the given prefix. O(n * m)"""
        return len(self.auto_complete(prefix))

    def is_empty(self) -> bool:
        """Check if the trie is empty. O(1)"""
        return self.word_count == 0

    def __len__(self) -> int:
        """Return the number of words in the trie. O(1)"""
        return self.word_count


class TrieNodeList:
    """A node in a dynamic-list Trie."""

    def __init__(self, char: str = "") -> None:
        self.char: str = char
        self.children: list[TrieNodeList] = []
        self.is_end_of_word: bool = False


class TrieList:
    """
    Dynamic-List Trie implementation.

    Each node has a dynamic list of child nodes.
    Similar to LCRS but uses Python list instead of sibling pointers.

    Time Complexity:
        - insert: O(m * k) where m is word length, k is avg children per node
        - search: O(m * k)
        - starts_with: O(m * k)
        - delete: O(m * k)

    Space Complexity: O(n * m) - dynamic allocation
    """

    def __init__(self) -> None:
        self.root: TrieNodeList = TrieNodeList()
        self.word_count: int = 0

    def _find_child(self, node: TrieNodeList, char: str) -> Optional[TrieNodeList]:
        """Find a child with the given character. O(k)"""
        for child in node.children:
            if child.char == char:
                return child
        return None

    def insert(self, word: str) -> None:
        """Insert a word into the trie. O(m * k)"""
        node = self.root
        for char in word:
            child = self._find_child(node, char)
            if child is None:
                child = TrieNodeList(char)
                node.children.append(child)
            node = child
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1

    def search(self, word: str) -> bool:
        """Search for an exact word in the trie. O(m * k)"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix. O(m * k)"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> Optional[TrieNodeList]:
        """Find the node corresponding to the last character of prefix."""
        node = self.root
        for char in prefix:
            child = self._find_child(node, char)
            if child is None:
                return None
            node = child
        return node

    def delete(self, word: str) -> bool:
        """Delete a word from the trie. O(m * k)"""
        if not word:
            return False
        self._delete_recursive(self.root, word, 0)
        return True

    def _has_children(self, node: TrieNodeList) -> bool:
        """Check if node has any children."""
        return len(node.children) > 0

    def _delete_recursive(self, node: TrieNodeList, word: str, index: int) -> bool:
        """Helper for delete operation."""
        if index == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            self.word_count -= 1
            return not self._has_children(node)

        char = word[index]
        child = self._find_child(node, char)
        if child is None:
            return False

        should_delete_child = self._delete_recursive(child, word, index + 1)

        if should_delete_child:
            node.children.remove(child)
            return not self._has_children(node) and not node.is_end_of_word

        return False

    def get_all_words(self) -> list[str]:
        """Get all words stored in the trie. O(n * m)"""
        words: list[str] = []
        self._collect_words(self.root, "", words)
        return words

    def _collect_words(
        self, node: TrieNodeList, prefix: str, words: list[str]
    ) -> None:
        """Helper to collect all words from a given node."""
        current_prefix = prefix
        if node.char:
            current_prefix = prefix + node.char

        if node.is_end_of_word:
            words.append(current_prefix)

        for child in node.children:
            self._collect_words(child, current_prefix, words)

    def auto_complete(self, prefix: str) -> list[str]:
        """Get all words that start with the given prefix. O(n * m)"""
        node = self._find_node(prefix)
        if node is None:
            return []
        words: list[str] = []
        if node.is_end_of_word:
            words.append(prefix)
        for child in node.children:
            self._collect_words(child, prefix, words)
        return words

    def count_words_with_prefix(self, prefix: str) -> int:
        """Count how many words start with the given prefix. O(n * m)"""
        return len(self.auto_complete(prefix))

    def is_empty(self) -> bool:
        """Check if the trie is empty. O(1)"""
        return self.word_count == 0

    def __len__(self) -> int:
        """Return the number of words in the trie. O(1)"""
        return self.word_count


def test_trie() -> None:
    """Test Trie (Dictionary-based) operations."""
    print("=" * 60)
    print("Testing Trie (Dictionary-based)")
    print("=" * 60)

    trie = Trie()

    # Test empty trie
    print("\n1. Empty trie:")
    print(f"   is_empty(): {trie.is_empty()}")
    print(f"   search('hello'): {trie.search('hello')}")
    print(f"   word_count: {len(trie)}")

    # Test insert
    print("\n2. Testing insert():")
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    for word in words:
        trie.insert(word)
        print(f"   insert('{word}'): word_count={len(trie)}")

    # Test search
    print("\n3. Testing search():")
    search_tests = ["apple", "app", "application", "appl", "orange", "ban"]
    for word in search_tests:
        print(f"   search('{word}'): {trie.search(word)}")

    # Test starts_with (prefix search)
    print("\n4. Testing starts_with():")
    prefix_tests = ["app", "appl", "ban", "xyz", "a"]
    for prefix in prefix_tests:
        print(f"   starts_with('{prefix}'): {trie.starts_with(prefix)}")

    # Test auto_complete (autocomplete)
    print("\n5. Testing auto_complete() - autocomplete:")
    autocomplete_tests = ["app", "band", "ban", "xyz"]
    for prefix in autocomplete_tests:
        results = trie.auto_complete(prefix)
        print(f"   auto_complete('{prefix}'): {results}")

    # Test count_words_with_prefix
    print("\n6. Testing count_words_with_prefix():")
    for prefix in ["app", "ban", "b"]:
        print(f"   count_words_with_prefix('{prefix}'): {trie.count_words_with_prefix(prefix)}")

    # Test get_all_words
    print("\n7. Testing get_all_words():")
    all_words = trie.get_all_words()
    print(f"   All words in trie: {all_words}")
    print(f"   Total word count: {len(trie)}")

    # Test delete
    print("\n8. Testing delete():")
    print(f"   Before delete - search('app'): {trie.search('app')}")
    print(f"   delete('app'): {trie.delete('app')}")
    print(f"   After delete - search('app'): {trie.search('app')}")
    print(f"   search('apple') still exists: {trie.search('apple')}")
    print(f"   search('application') still exists: {trie.search('application')}")
    print(f"   Word count after delete: {len(trie)}")

    # Test words that are prefixes of each other
    print("\n9. Testing prefix chain (the -> them -> theme -> themes):")
    prefix_trie = Trie()
    chain_words = ["the", "them", "theme", "themes"]
    for word in chain_words:
        prefix_trie.insert(word)
        print(f"   insert('{word}')")
    print(f"   All words: {prefix_trie.get_all_words()}")
    print(f"   delete('them'): {prefix_trie.delete('them')}")
    print(f"   After delete: {prefix_trie.get_all_words()}")
    print(f"   search('the'): {prefix_trie.search('the')}")
    print(f"   search('theme'): {prefix_trie.search('theme')}")

    # Test duplicate insertion
    print("\n10. Testing duplicate insertion:")
    dup_trie = Trie()
    dup_trie.insert("hello")
    print(f"   After insert('hello'): count={len(dup_trie)}")
    dup_trie.insert("hello")
    print(f"   After duplicate insert('hello'): count={len(dup_trie)}")

    # Time complexity summary
    print("\n" + "=" * 60)
    print("Time Complexity Summary")
    print("=" * 60)
    print(f"\n{'Operation':<30} {'Time':<15} {'Description':<25}")
    print(f"{'-' * 30} {'-' * 15} {'-' * 25}")
    print(f"{'insert(word)':<30} {'O(m)':<15} {'m = word length':<25}")
    print(f"{'search(word)':<30} {'O(m)':<15} {'Exact match':<25}")
    print(f"{'starts_with(prefix)':<30} {'O(m)':<15} {'Prefix exists check':<25}")
    print(f"{'delete(word)':<30} {'O(m)':<15} {'Remove word':<25}")
    print(f"{'auto_complete()':<30} {'O(n * m)':<15} {'Autocomplete':<25}")
    print(f"{'get_all_words()':<30} {'O(n * m)':<15} {'n = total words':<25}")


def test_trie_lcrs() -> None:
    """Test TrieLCRS (Left-Child Right-Sibling) operations."""
    print("\n" + "=" * 60)
    print("Testing TrieLCRS (Left-Child Right-Sibling)")
    print("=" * 60)

    trie = TrieLCRS()

    # Test empty trie
    print("\n1. Empty trie:")
    print(f"   is_empty(): {trie.is_empty()}")
    print(f"   search('hello'): {trie.search('hello')}")
    print(f"   word_count: {len(trie)}")

    # Test insert
    print("\n2. Testing insert():")
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    for word in words:
        trie.insert(word)
        print(f"   insert('{word}'): word_count={len(trie)}")

    # Test search
    print("\n3. Testing search():")
    search_tests = ["apple", "app", "application", "appl", "orange", "ban"]
    for word in search_tests:
        print(f"   search('{word}'): {trie.search(word)}")

    # Test starts_with
    print("\n4. Testing starts_with():")
    prefix_tests = ["app", "appl", "ban", "xyz", "a"]
    for prefix in prefix_tests:
        print(f"   starts_with('{prefix}'): {trie.starts_with(prefix)}")

    # Test auto_complete (autocomplete)
    print("\n5. Testing auto_complete() - autocomplete:")
    autocomplete_tests = ["app", "band", "ban", "xyz"]
    for prefix in autocomplete_tests:
        results = trie.auto_complete(prefix)
        print(f"   auto_complete('{prefix}'): {results}")

    # Test get_all_words
    print("\n6. Testing get_all_words():")
    all_words = trie.get_all_words()
    print(f"   All words in trie: {all_words}")

    # Test delete
    print("\n7. Testing delete():")
    print(f"   Before delete - search('app'): {trie.search('app')}")
    print(f"   delete('app'): {trie.delete('app')}")
    print(f"   After delete - search('app'): {trie.search('app')}")
    print(f"   search('apple') still exists: {trie.search('apple')}")
    print(f"   Word count after delete: {len(trie)}")


def test_trie_array() -> None:
    """Test TrieArray (Fixed-Array) operations."""
    print("\n" + "=" * 60)
    print("Testing TrieArray (Fixed-Array, a-z only)")
    print("=" * 60)

    trie = TrieArray()

    # Test empty trie
    print("\n1. Empty trie:")
    print(f"   is_empty(): {trie.is_empty()}")
    print(f"   search('hello'): {trie.search('hello')}")

    # Test insert
    print("\n2. Testing insert():")
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    for word in words:
        trie.insert(word)
        print(f"   insert('{word}'): word_count={len(trie)}")

    # Test search
    print("\n3. Testing search():")
    for word in ["apple", "app", "appl", "orange"]:
        print(f"   search('{word}'): {trie.search(word)}")

    # Test autocomplete
    print("\n4. Testing auto_complete():")
    for prefix in ["app", "ban"]:
        print(f"   auto_complete('{prefix}'): {trie.auto_complete(prefix)}")

    # Test delete
    print("\n5. Testing delete():")
    print(f"   delete('app'): {trie.delete('app')}")
    print(f"   search('app'): {trie.search('app')}")
    print(f"   search('apple'): {trie.search('apple')}")


def test_trie_list() -> None:
    """Test TrieList (Dynamic-List) operations."""
    print("\n" + "=" * 60)
    print("Testing TrieList (Dynamic-List)")
    print("=" * 60)

    trie = TrieList()

    # Test empty trie
    print("\n1. Empty trie:")
    print(f"   is_empty(): {trie.is_empty()}")
    print(f"   search('hello'): {trie.search('hello')}")

    # Test insert
    print("\n2. Testing insert():")
    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]
    for word in words:
        trie.insert(word)
        print(f"   insert('{word}'): word_count={len(trie)}")

    # Test search
    print("\n3. Testing search():")
    for word in ["apple", "app", "appl", "orange"]:
        print(f"   search('{word}'): {trie.search(word)}")

    # Test autocomplete
    print("\n4. Testing auto_complete():")
    for prefix in ["app", "ban"]:
        print(f"   auto_complete('{prefix}'): {trie.auto_complete(prefix)}")

    # Test delete
    print("\n5. Testing delete():")
    print(f"   delete('app'): {trie.delete('app')}")
    print(f"   search('app'): {trie.search('app')}")
    print(f"   search('apple'): {trie.search('apple')}")


def test_trie_comparison() -> None:
    """Compare all Trie implementations."""
    print("\n" + "=" * 60)
    print("All Trie Implementations Comparison")
    print("=" * 60)

    words = ["apple", "app", "application", "apply", "banana", "band", "bandana"]

    # Create all tries
    dict_trie = Trie()
    lcrs_trie = TrieLCRS()
    array_trie = TrieArray()
    list_trie = TrieList()

    for word in words:
        dict_trie.insert(word)
        lcrs_trie.insert(word)
        array_trie.insert(word)
        list_trie.insert(word)

    print(f"\nInserted words: {words}")

    # Compare search results
    print("\n1. Search results:")
    for word in ["apple", "app", "appl", "xyz"]:
        results = [
            dict_trie.search(word),
            lcrs_trie.search(word),
            array_trie.search(word),
            list_trie.search(word),
        ]
        all_match = len(set(results)) == 1
        status = "OK" if all_match else "MISMATCH"
        print(f"   '{word}': Dict={results[0]}, LCRS={results[1]}, Array={results[2]}, List={results[3]} [{status}]")

    # Compare autocomplete
    print("\n2. Autocomplete for 'app':")
    dict_result = sorted(dict_trie.auto_complete("app"))
    lcrs_result = sorted(lcrs_trie.auto_complete("app"))
    array_result = sorted(array_trie.auto_complete("app"))
    list_result = sorted(list_trie.auto_complete("app"))
    print(f"   Dict:  {dict_result}")
    print(f"   LCRS:  {lcrs_result}")
    print(f"   Array: {array_result}")
    print(f"   List:  {list_result}")

    # Compare after delete
    print("\n3. After delete('app'):")
    dict_trie.delete("app")
    lcrs_trie.delete("app")
    array_trie.delete("app")
    list_trie.delete("app")
    print(f"   Dict:  {sorted(dict_trie.get_all_words())}")
    print(f"   LCRS:  {sorted(lcrs_trie.get_all_words())}")
    print(f"   Array: {sorted(array_trie.get_all_words())}")
    print(f"   List:  {sorted(list_trie.get_all_words())}")

    # Implementation comparison table
    print("\n" + "=" * 60)
    print("Implementation Comparison")
    print("=" * 60)
    print(f"\n{'Aspect':<18} {'Dict':<14} {'LCRS':<14} {'Array':<14} {'List':<14}")
    print(f"{'-' * 18} {'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14}")
    print(f"{'Node storage':<18} {'dict[c,Node]':<14} {'child Node+sibling Node':<14} {'[None]*26':<14} {'list[Node]':<14}")
    print(f"{'Child lookup':<18} {'O(1) hash':<14} {'O(k) scan':<14} {'O(1) index':<14} {'O(k) scan':<14}")
    print(f"{'Memory/node':<18} {'Dynamic':<14} {'Minimal':<14} {'Fixed 26':<14} {'Dynamic':<14}")
    print(f"{'Insert time':<18} {'O(m)':<14} {'O(m*k)':<14} {'O(m)':<14} {'O(m*k)':<14}")
    print(f"{'Char support':<18} {'Any':<14} {'Any':<14} {'a-z only':<14} {'Any':<14}")

    print("\n" + "=" * 60)
    print("When to use which?")
    print("=" * 60)
    print("\nTrie (Dictionary):")
    print("   - Best general-purpose choice")
    print("   - O(1) child lookup, flexible character set")
    print("\nTrieLCRS (Left-Child Right-Sibling):")
    print("   - Minimal memory usage")
    print("   - Good for learning tree representations")
    print("\nTrieArray (Fixed Array):")
    print("   - Fastest lookup (direct index)")
    print("   - Best when alphabet is fixed (a-z)")
    print("\nTrieList (Dynamic List):")
    print("   - Simpler than LCRS, similar performance")
    print("   - Good middle ground for sparse tries")
