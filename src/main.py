from linked_list.linked_list import (
    test_linked_list_head_only,
    test_linked_list_head_tail,
    test_doubly_linked_list,
    test_comparison,
)
from stack_queue.stack import test_stack
from stack_queue.queue import test_queue
from trees.binary_tree import test_binary_search_tree


if __name__ == "__main__":
    test_linked_list_head_only()
    test_linked_list_head_tail()
    test_doubly_linked_list()
    test_comparison()
    test_stack()
    test_queue()
    test_binary_search_tree()
