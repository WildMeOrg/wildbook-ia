from __future__ import absolute_import, division, print_function
cimport cython
import cython


CYTHONIZED = True


cdef class TreeNode:
    cdef long id_
    cdef long level
    cdef list child_nodes
    cdef TreeNode parent_node

    def __init__(self, long id_, TreeNode parent_node, long level):
        self.id_ = id_
        self.parent_node = parent_node
        self.child_nodes = []
        self.level = level

    cpdef set_children(self, list child_nodes):
        self.child_nodes = child_nodes

    cpdef list get_children(self):
        return self.child_nodes

    def __getitem__(self, long index):
        return self.child_nodes[index]

    cpdef get_parent(self):
        return self.parent_node

    cpdef long get_num_children(self):
        return len(self.child_nodes)

    cpdef long get_id(self):
        return self.id_

    cpdef long get_row(self):
        cdef list sibling_nodes = self.parent_node.child_nodes
        cdef long row = sibling_nodes.index(self)
        return row

    cpdef long get_level(self):
        return self.level


#@cython.boundscheck(False)
#@cython.wraparound(False)
def _populate_tree_recursive(TreeNode parent_node,
                             list child_ids,
                             long num_levels,
                             list ider_list,
                             long level):
    """ Recursively builds the tree structure """
    if level == num_levels - 1:
        child_nodes = [TreeNode(id_, parent_node, level) for id_ in child_ids]
    else:
        child_ider = ider_list[level + 1]
        child_nodes =  [_populate_tree_recursive(
            TreeNode(id_, parent_node, level),
            child_ider(id_),
            num_levels,
            ider_list,
            level + 1)
            for id_ in child_ids]
    parent_node.child_nodes = child_nodes
    return parent_node


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef _populate_tree_iterative(TreeNode root_node, long num_levels, list ider_list):
    """ Iteratively builds the tree structure. I dont quite trust this yet """
    cdef TreeNode parent_node
    cdef size_t level
    cdef size_t id_
    cdef list root_ids
    cdef list parent_node_list
    cdef list ids_list
    cdef list next_ids
    cdef list new_node_lists 
    cdef list new_ids_lists  

    root_ids = ider_list[0]()
    ids_list = [root_ids]
    parent_node_list = [root_node]
    for level in range(num_levels):
        #print('------------ level=%r -----------' % (level,))
        #print(utool.dict_str(locals()))
        new_node_lists = []
        new_ids_lists  = []
        for parent_node, id_list in zip(parent_node_list, ids_list):
            #pass
            #assert isinstance(parent_node, TreeNode), '%r\n%s' % (parent_node,
            #                                                      utool.dict_str(locals()))
            node_list =  [TreeNode(id_, parent_node, level) for id_ in id_list]
            if level + 1 < num_levels:
                child_ider = ider_list[level + 1]
                next_ids =  child_ider(id_list)
                #[child_ider(id_) for id_ in child_ids]
            else:
                next_ids = []
            parent_node.child_nodes = node_list
            new_node_lists.extend(node_list)
            new_ids_lists.extend(next_ids)
        parent_node_list = new_node_lists
        ids_list = new_ids_lists


#@cython.boundscheck(False)
#@cython.wraparound(False)
def build_internal_structure(model):
    #from guitool.api_item_model import *
    ider_list = model.iders
    num_levels = len(ider_list)
    USE_RECURSIVE = True
    if USE_RECURSIVE:
        # I trust this code more although it is slightly slower
        if num_levels == 0:
            root_id_list = []
        else:
            root_id_list = ider_list[0]()
        root_node = TreeNode(-1, None, -1)
        level = 0
        _populate_tree_recursive(root_node, root_id_list, num_levels, ider_list, level)
    else:
        # TODO: Vet this code a bit more.
        root_node = TreeNode(-1, None, -1)
        _populate_tree_iterative(root_node, num_levels, ider_list)
    #print(root_node.full_str())
    #assert root_node.__dict__, "root_node.__dict__ is empty"
    return root_node
