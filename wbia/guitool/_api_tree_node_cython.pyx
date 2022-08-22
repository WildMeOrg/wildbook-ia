from __future__ import absolute_import, division, print_function

cimport cython

import cython

#from libc.stdlib cimport malloc, free  # TODO: convert lists to arrays


global CYTHONIZED
CYTHONIZED = True


cdef class TreeNode:
    cdef long id_
    cdef long level
    cdef list child_nodes
    cdef TreeNode parent_node

    def __init__(TreeNode self, long id_, TreeNode parent_node, long level):
        self.id_ = id_
        self.parent_node = parent_node
        self.child_nodes = []
        self.level = level

    def __getitem__(self, long index):
        return self.get_child(index)

    cpdef set_children(TreeNode self, list child_nodes):
        self.child_nodes = child_nodes

    cpdef list get_children(TreeNode self):
        return self.child_nodes

    cpdef TreeNode get_child(self, long index):
        return self.child_nodes[index]

    cpdef get_parent(TreeNode self):
        return self.parent_node

    cpdef long child_index(self, TreeNode child_node):
        return self.child_nodes.index(child_node)

    cpdef long get_num_children(TreeNode self):
        return len(self.child_nodes)

    cpdef long get_id(TreeNode self):
        return self.id_

    cpdef long get_row(TreeNode self):
        cdef list sibling_nodes = self.parent_node.child_nodes
        cdef long row = sibling_nodes.index(self)
        return row

    cpdef long get_level(TreeNode self):
        return self.level


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef TreeNode _populate_tree_recursive(TreeNode parent_node,
                             list child_ids,
                             long num_levels,
                             list ider_list,
                             long level):
    """ Recursively builds the tree structure """
    cdef:
        size_t ix
        long id_
        list child_nodes
        TreeNode next_node
        list next_ids
    if level == num_levels - 1:
        #def TreeNode* child_nodes = malloc(sizeof(TreeNode) * len(child_ids))
        child_nodes = [None] * len(child_ids)
        for ix in range(len(child_ids)):
            id_ = child_ids[ix]
            child_nodes[ix] = TreeNode(id_, parent_node, level)
        #child_nodes = [TreeNode(id_, parent_node, level) for id_ in child_ids]
    else:
        child_nodes = [None] * len(child_ids)
        for ix in range(len(child_ids)):
            id_ = child_ids[ix]
            next_ids = ider_list[level + 1](id_)
            next_node = TreeNode(id_, parent_node, level)
            child_nodes[ix] = _populate_tree_recursive(next_node, next_ids, num_levels, ider_list, level + 1)
        #child_nodes =  [ for id_ in child_ids]
    parent_node.child_nodes = child_nodes
    return parent_node


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _populate_tree_iterative(TreeNode root_node, long num_levels, list ider_list):
    """ Iteratively builds the tree structure. I dont quite trust this yet """
    cdef:
        TreeNode parent_node
        size_t level
        size_t ix
        size_t sx
        long id_
        list root_ids
        list parent_node_list
        list ids_list
        list id_list
        list next_ids
        list node_list
        list new_node_lists
        list new_ids_lists

    root_ids = ider_list[0]()
    ids_list = [root_ids]
    parent_node_list = [root_node]
    for level in range(num_levels):
        #print('------------ level=%r -----------' % (level,))
        #print(utool.repr2(locals()))
        new_node_lists = []
        new_ids_lists  = []
        for sx in range(len(ids_list)):
            parent_node = parent_node_list[sx]
            id_list = ids_list[sx]
            node_list = [None] * len(id_list)
            #node_list =  [TreeNode(id_, parent_node, level) for id_ in id_list]
            for ix, id_ in enumerate(id_list):
                node_list[ix] = TreeNode(id_, parent_node, level)
            if level + 1 < num_levels:
                next_ids =  ider_list[level + 1](id_list)
                #[child_ider(id_) for id_ in child_ids]
            else:
                next_ids = []
            parent_node.child_nodes = node_list
            new_node_lists.extend(node_list)
            new_ids_lists.extend(next_ids)
        parent_node_list = new_node_lists
        ids_list = new_ids_lists


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef TreeNode build_internal_structure(object model):
    #from wbia.guitool.api_item_model import *
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
        #with nogil:
        _populate_tree_recursive(root_node, root_id_list, num_levels, ider_list, level)
    else:
        # TODO: Vet this code a bit more.
        root_node = TreeNode(-1, None, -1)
        #with nogil:
        _populate_tree_iterative(root_node, num_levels, ider_list)
    #print(root_node.full_str())
    #assert root_node.__dict__, "root_node.__dict__ is empty"
    return root_node
