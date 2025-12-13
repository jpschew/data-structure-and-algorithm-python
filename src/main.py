from linked_list.linked_list import (
    test_linked_list_head_only,
    test_linked_list_head_tail,
    test_doubly_linked_list,
    test_comparison,
)
from stack_queue.stack import test_stack
from stack_queue.queue import test_queue, test_queue_using_stacks, test_priority_queues
from trees.binary_tree import test_binary_search_tree
from trees.avl_tree import test_avl_tree
from heaps.binary_heap import (
    test_min_heap,
    test_max_heap,
    test_heap_comparison,
    test_priority_queue_heap,
)
from tries.trie import (
    test_trie,
    test_trie_lcrs,
    test_trie_array,
    test_trie_list,
    test_trie_comparison,
)


if __name__ == "__main__":
    test_linked_list_head_only()
    test_linked_list_head_tail()
    test_doubly_linked_list()
    test_comparison()
    test_stack()
    test_queue()
    test_queue_using_stacks()
    test_priority_queues()
    test_binary_search_tree()
    test_avl_tree()
    test_min_heap()
    test_max_heap()
    test_heap_comparison()
    test_priority_queue_heap()
    test_trie()
    test_trie_lcrs()
    test_trie_array()
    test_trie_list()
    test_trie_comparison()
