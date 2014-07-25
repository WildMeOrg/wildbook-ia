from __future__ import absolute_import, division, print_function
import cython
ctypedef int ITYPE_t


cdef class TreeNode():  # (object):
    # USE THIS IN CONTROLLER
    # Available in Python-space:
    #property period:
    #    def __get__(self):
    #        return 1.0 / self.freq
    #    def __set__(self, value):
    #        self.freq = 1.0 / value

    #__slots__ = ('id_', 'parent_node', 'child_nodes', 'level',)
    def __init__(self, int id_, parent_node, level):
        cdef int self.id_ = id_
        TreeNode self.parent_node = parent_node
        self.child_nodes = []
        self.level = level

    def __del__(self):
        if utool.VERBOSE:
            print('[guitool] DELETING THE TREE NODE!: id_=%r' % self.id_)

    def set_children(self, child_nodes):
        self.child_nodes = child_nodes

    def get_children(self):
        return self.child_nodes

    def __getitem__(self, index):
        return self.child_nodes[index]

    def get_parent(self):
        return self.parent_node

    def get_num_children(self):
        return len(self.child_nodes)

    def get_id(self):
        return self.id_

    def get_row(self):
        sibling_nodes = self.parent_node.child_nodes
        row = sibling_nodes.index(self)
        return row

    def get_level(self):
        return self.level

    def full_str(self, indent=""):
        self_str = indent + "TreeNode(id_=%r, parent_node=%r, level=%r)" % (self.id_, id(self.parent_node), self.level)
        child_strs = [child.full_str(indent=indent + "    ") for child in self.child_nodes]
        str_ = "\n".join([self_str] + child_strs)
        return str_

@cython.nonecheck(False)
@profile
def _populate_tree_recursive(parent_node, child_ids, num_levels, ider_list, level):
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
    parent_node.set_children(child_nodes)
    return parent_node


@profile
def _populate_tree_iterative(root_node, num_levels, ider_list):
    """ Iteratively builds the tree structure. I dont quite trust this yet """
    root_ids = ider_list[0]()
    parent_node_list = [root_node]
    ids_list = [root_ids]
    for level in xrange(num_levels):
        #print('------------ level=%r -----------' % (level,))
        #print(utool.dict_str(locals()))
        new_node_lists = []
        new_ids_lists  = []
        for parent_node, id_list in izip(parent_node_list, ids_list):
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
            parent_node.set_children(node_list)
            new_node_lists.extend(node_list)
            new_ids_lists.extend(next_ids)
        parent_node_list = new_node_lists
        ids_list = new_ids_lists


@profile
def _build_internal_structure(model):
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
        root_node = TreeNode(None, None, -1)
        level = 0
        _populate_tree_recursive(root_node, root_id_list, num_levels, ider_list, level)
    else:
        # TODO: Vet this code a bit more.
        root_node = TreeNode(None, None, -1)
        _populate_tree_iterative(root_node, num_levels, ider_list)
    #print(root_node.full_str())
    #assert root_node.__dict__, "root_node.__dict__ is empty"
    return root_node
