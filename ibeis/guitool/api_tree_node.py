# TODO: Rename api_item_model
from __future__ import absolute_import, division, print_function
import utool
from itertools import izip
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[tree_node]', DEBUG=False)


CYTHONIZED = False


class TreeNode(object):
    # USE THIS IN CONTROLLER
    # Available in Python-space:
    #property period:
    #    def __get__(self):
    #        return 1.0 / self.freq
    #    def __set__(self, value):
    #        self.freq = 1.0 / value

    __slots__ = ('id_', 'parent_node', 'child_nodes', 'level',)
    def __init__(self, id_, parent_node, level):
        self.id_ = id_
        self.parent_node = parent_node
        self.child_nodes = []
        self.level = level

    #def __del__(self):
    #    if utool.VERBOSE:
    #        print('[guitool] DELETING THE TREE NODE!: id_=%r' % self.id_)

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


def tree_node_string(self, indent='', charids=True, id_dict={}, last=['A']):
    id_ = self.get_id()
    level = self.get_level()
    id_self = id(self)
    id_parent = id(self.get_parent())
    if charids:
        if id_parent not in id_dict:
            id_dict[id_parent] = last[0]
            last[0] = chr(ord(last[0]) + 1)
        if id_self not in id_dict:
            id_dict[id_self] = last[0]
            last[0] = chr(ord(last[0]) + 1)
        id_self = id_dict[id_self]
        id_parent = id_dict[id_parent]
    tup = (id_, level, str(id_self), str(id_parent))
    self_str = (indent + "TreeNode(id_=%r, level=%r, self=%s, parent_node=%s)" % tup)
    child_strs = [tree_node_string(child, indent=indent + '    ', charids=charids, id_dict=id_dict, last=last) for child in self.get_children()]
    str_ = '\n'.join([self_str] + child_strs)
    return str_


#@profile
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


#try:
#    #from . import api_tree_node_cython  # NOQA
#    #TreeNode = api_tree_node_cython.TreeNode  # NOQA
#    print('[guitool] Using Cython TreeNode')
#except ImportError:
#    print('[guitool] Using Python TreeNode')
