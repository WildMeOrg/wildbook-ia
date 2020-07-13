# -*- coding: utf-8 -*-
# TODO: Rename api_item_model
from __future__ import absolute_import, division, print_function
from wbia.guitool.__PYQT__ import QtCore  # NOQA
from types import GeneratorType
from six.moves import zip, range
import utool
import utool as ut

(print, print_, rrr) = utool.inject2(__name__)


TREE_NODE_BASE = QtCore.QObject
# TREE_NODE_BASE = object
VERBOSE_TREE_NODE = ut.get_argflag(('--verb-qt-tree'))


class TreeNode(TREE_NODE_BASE):
    """
    Cyth:
        cdef:
            long id_, level
            list child_nodes
            TreeNode parent_node
    """

    # __slots__ = ('id_', 'parent_node', 'child_nodes', 'level',)
    def __init__(self, id_, parent_node, level):
        TREE_NODE_BASE.__init__(self, parent=parent_node)
        # super(TreeNode, self).__init__(parent_node)
        # if TREE_NODE_BASE is not object:
        # if VERBOSE_TREE_NODE:
        #    print('[TreeNode] __init__')
        # super(TreeNode, self).__init__(parent=parent_node)
        self.id_ = id_
        self.parent_node = parent_node
        self.child_nodes = []
        self.level = level

    def __del__(self):
        # print('[guitool] DELETING THE TREE NODE!:')
        if VERBOSE_TREE_NODE:
            print('[guitool] DELETING THE TREE NODE!: id_=%r' % self.id_)

    def __getitem__(self, index):
        """
        <CYTH returns="TreeNode">
        cdef long index
         """
        return self.get_child(index)

    def set_children(self, child_nodes):
        """ <CYTH returns="void"> """
        self.child_nodes = child_nodes

    def get_children(self):
        """ </CYTH returns="list"> """
        self.lazy_checks()
        return self.child_nodes

    def child_index(self, child_node):
        """
        <CYTH returns=long>
        cdef TreeNode child_node

        """
        self.lazy_checks()
        return self.child_nodes.index(child_node)

    def get_child(self, index):
        """
        <CYTH returns="TreeNode">
        cdef long index
         """
        self.lazy_checks()
        return self.child_nodes[index]

    def get_parent(self):
        """ <CYTH returns="TreeNode"> """
        try:
            return self.parent_node
        except AttributeError as ex:
            import utool as ut

            ut.printex(ex, 'Error getting parent', tb=True)
            # print(ex)
            # print('[tree_node] dir(self)=')
            # print(dir(self))
            # print('[tree_node] self.__dict__=')
            # print(utool.repr2(self.__dict__))
            raise

    def get_num_children(self):
        """ <CYTH returns=long>
        """
        self.lazy_checks()
        return len(self.child_nodes)

    def get_id(self):
        """ Returns python internal id of this class
        <CYTH returns="long"> """
        return self.id_

    def get_row(self):
        """ Returns the row_index of this node w.r.t its parent.

        cdef list sibling_nodes
        cdef long row

        """
        sibling_nodes = self.parent_node.child_nodes
        row = sibling_nodes.index(self)
        return row

    def get_level(self):
        """ <CYTH returns="long"> """
        return self.level

    def find_row_from_id(self, _id):
        """
        given an id (like an wbia rowid) find the row of this item
        """
        children = self.get_children()
        id_list = [child.get_id() for child in children]
        row = ut.listfind(id_list, _id)
        return row

    def lazy_checks(self):
        # If the child is a generator, then the TreeNode hasn't been created yet
        # so create it
        if isinstance(self.child_nodes, GeneratorType):
            print('[tree_node] lazy evaluation level=%r' % self.level)
            # print('[tree_node] lazy evaluation level=%r' % self.level)
            self.child_nodes = list(self.child_nodes)


def tree_node_string(self, indent='', charids=True, id_dict=None, last=None):
    """ makes a recrusive string representation of a treee

    HACK:if charids is 2  uses ordinals instead of characters
    if charirsd is True triesto use Numbers
    otherwise uses nondetermensitic python ids
    """
    if last is None:
        if charids == 2:
            last = [0]
        else:
            last = ['A']
    if id_dict is None:
        id_dict = {}
    id_ = self.get_id()
    level = self.get_level()
    id_self = id(self)
    id_parent = id(self.get_parent())
    if charids == 2:
        if id_parent not in id_dict:
            id_dict[id_parent] = last[0]
            last[0] = last[0] + 1
        if id_self not in id_dict:
            id_dict[id_self] = last[0]
            last[0] = last[0] + 1
        id_self = id_dict[id_self]
        id_parent = id_dict[id_parent]
    elif charids is True:
        # if ord(last[0]) < 255:
        #    last[0] = [0]
        if id_parent not in id_dict:
            id_dict[id_parent] = last[0]
            last[0] = chr(ord(last[0]) + 1)
        if id_self not in id_dict:
            id_dict[id_self] = last[0]
            last[0] = chr(ord(last[0]) + 1)
        id_self = id_dict[id_self]
        id_parent = id_dict[id_parent]
    tup = (id_, level, str(id_self), str(id_parent))
    self_str = indent + 'TreeNode(id_=%r, level=%r, self=%s, parent_node=%s)' % tup
    child_strs = [
        tree_node_string(
            child, indent=indent + '    ', charids=charids, id_dict=id_dict, last=last
        )
        for child in self.get_children()
    ]
    str_ = '\n'.join([self_str] + child_strs)
    return str_


def _populate_tree_iterative(root_node, num_levels, ider_list):
    """ Iteratively builds the tree structure. I dont quite trust this yet
    #@cython.boundscheck(False)
    #@cython.wraparound(False)

    Cyth::
        @returns(TreeNode)
        <CYTH returns="TreeNode">
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

    Args:
        root_node (?):
        num_levels (?):
        ider_list (list):

    CommandLine:
        python -m wbia.guitool.api_tree_node --test-_populate_tree_iterative

    Example:
        >>> # xdoctest: +REQUIRES(module:wbia)
        >>> from wbia.guitool.api_tree_node import *  # NOQA
        >>> import utool as ut
        >>> from wbia.guitool import api_tree_node  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(ut.get_argval('--db', str, default='testdb1'))
        >>> # build test data
        >>> ider_list = [ibs.get_valid_nids, ibs.get_name_aids]
        >>> num_levels = len(ider_list)
        >>> # execute function
        >>> root_node = TreeNode(-1, None, -1)
        >>> api_tree_node._populate_tree_iterative(root_node, num_levels, ider_list)
        >>> # verify results
        >>> self = root_node
        >>> infostr = api_tree_node.tree_node_string(root_node, charids=2)
        >>> # print(ut.truncate_str(infostr, maxlen=2000))
        >>> result = ut.hashstr(infostr)
        >>> print(result)
    """
    root_ids = ider_list[0]()
    parent_node_list = [root_node]
    ids_list = [root_ids]
    if VERBOSE_TREE_NODE:
        print('_populate_tree_iterative')
        print('root_ids = %r' % (root_ids,))
        print('num_levels = %r' % (num_levels,))
    for level in range(num_levels):
        # print('------------ level=%r -----------' % (level,))
        # print(utool.repr2(locals()))
        new_node_lists = []
        new_ids_lists = []
        for parent_node, id_list in zip(parent_node_list, ids_list):
            # pass
            # assert isinstance(parent_node, TreeNode), '%r\n%s' % (parent_node,
            #                                                      utool.repr2(locals()))
            node_list = [TreeNode(id_, parent_node, level) for id_ in id_list]
            if level + 1 < num_levels:
                child_ider = ider_list[level + 1]
                next_ids = child_ider(id_list)
                # [child_ider(id_) for id_ in child_ids]
            else:
                next_ids = []
            parent_node.set_children(node_list)
            new_node_lists.extend(node_list)
            new_ids_lists.extend(next_ids)
        parent_node_list = new_node_lists
        ids_list = new_ids_lists


def _populate_tree_recursive(parent_node, child_ids, num_levels, ider_list, level):
    """
    Recursively builds the tree structure

    Cyth::
        <CYTH returns="TreeNode">
        cdef:
            size_t ix
            long id_
            list child_nodes
            TreeNode next_node
            list next_ids

    Example:
        >>> # xdoctest: +REQUIRES(module:wbia)
        >>> from wbia.guitool.api_tree_node import *  # NOQA
        >>> from wbia.guitool.api_tree_node import *  # NOQA
        >>> import utool as ut
        >>> from wbia.guitool import api_tree_node  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(ut.get_argval('--db', str, default='testdb1'))
        >>> # build test data
        >>> ider_list = [ibs.get_valid_nids, ibs.get_name_aids]
        >>> num_levels = len(ider_list)
        >>> root_node = TreeNode(-1, None, -1)
        >>> if num_levels == 0:
        >>>     root_id_list = []
        >>> else:
        >>>     root_id_list = ider_list[0]()
        >>> root_node = TreeNode(-1, None, -1)
        >>> level = 0
        >>> # execute function
        >>> api_tree_node._populate_tree_recursive(root_node, root_id_list, num_levels, ider_list, level)
        >>> # verify results
        >>> self = root_node
        >>> infostr = api_tree_node.tree_node_string(root_node, charids=2)
        >>> # print(ut.truncate_str(infostr, maxlen=2000))
        >>> result = ut.hashstr(infostr)
        >>> print(result)
    """
    if level == num_levels - 1:
        child_nodes = (TreeNode(id_, parent_node, level) for id_ in child_ids)
    else:
        child_ider = ider_list[level + 1]
        child_nodes = [
            _populate_tree_recursive(
                TreeNode(id_, parent_node, level),
                child_ider(id_),
                num_levels,
                ider_list,
                level + 1,
            )
            for id_ in child_ids
        ]
    parent_node.set_children(child_nodes)
    return parent_node


def _populate_tree_recursive_lazy(parent_node, child_ids, num_levels, ider_list, level):
    """
    Recursively builds the tree structure

    Cyth::
        <CYTH returns="TreeNode">
        cdef:
            size_t ix
            long id_
            list child_nodes
            TreeNode next_node
            list next_ids

    Example:
        >>> # xdoctest: +REQUIRES(module:wbia)
        >>> from wbia.guitool.api_tree_node import *  # NOQA
        >>> import utool as ut
        >>> from wbia.guitool import api_tree_node  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(ut.get_argval('--db', str, default='testdb1'))
        >>> # build test data
        >>> ider_list = [ibs.get_valid_nids, ibs.get_name_aids]
        >>> num_levels = len(ider_list)
        >>> root_node = TreeNode(-1, None, -1)
        >>> if num_levels == 0:
        >>>     root_id_list = []
        >>> else:
        >>>     root_id_list = ider_list[0]()
        >>> root_node = TreeNode(-1, None, -1)
        >>> level = 0
        >>> # execute function
        >>> api_tree_node._populate_tree_recursive_lazy(root_node, root_id_list, num_levels, ider_list, level)
        >>> # verify results
        >>> self = root_node
        >>> infostr = api_tree_node.tree_node_string(root_node, charids=2)
        >>> # print(ut.truncate_str(infostr, maxlen=2000))
        >>> result = ut.hashstr(infostr)
        >>> print(result)

    """
    if level == num_levels - 1:
        child_nodes_iter = (TreeNode(id_, parent_node, level) for id_ in child_ids)
    else:
        child_ider = ider_list[level + 1]
        child_nodes_iter = (
            _populate_tree_recursive(
                TreeNode(id_, parent_node, level),
                child_ider(id_),
                num_levels,
                ider_list,
                level + 1,
            )
            for id_ in child_ids
        )
    # seting children as an iterator triggers lazy loading
    parent_node.set_children(child_nodes_iter)
    return parent_node


def build_internal_structure(model):
    """
    Cyth:
        <CYTH returns="TreeNode">
    """
    # from wbia.guitool.api_item_model import *
    ider_list = model.iders  # an ider for each level
    ider_list = model.get_iders()
    num_levels = len(ider_list)
    # USE_RECURSIVE = True
    USE_RECURSIVE = False
    if USE_RECURSIVE:
        # I trust this code more although it is slightly slower
        if num_levels == 0:
            root_id_list = []
        else:
            root_id_list = ider_list[0]()
        root_node = TreeNode(-1, None, -1)
        level = 0
        # _populate_tree_recursive(root_node, root_id_list, num_levels, ider_list, level)
        _populate_tree_recursive_lazy(
            root_node, root_id_list, num_levels, ider_list, level
        )
    else:
        # TODO: Vet this code a bit more.
        root_node = TreeNode(-1, None, -1)
        _populate_tree_iterative(root_node, num_levels, ider_list)

    if VERBOSE_TREE_NODE:
        print('ider_list = %r' % (ider_list,))
        infostr = tree_node_string(root_node, charids=2)
        print(infostr)
        # print(ut.repr3(root_node.__dict__))
    # assert root_node.__dict__, "root_node.__dict__ is empty"
    return root_node


def build_scope_hack_list(root_node, scope_hack_list=[]):
    scope_hack_list.append(root_node)
    for child in root_node.get_children():
        build_scope_hack_list(child, scope_hack_list)


CYTHONIZED = False


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.guitool.api_tree_node
        python -m wbia.guitool.api_tree_node --allexamples
        python -m wbia.guitool.api_tree_node --allexamples --noface --nosrc
        python -m wbia.guitool.api_tree_node --allexamples --noface --nosrc --db GZ_ALL
        python -m wbia.guitool.api_tree_node --allexamples --noface --nosrc --db PZ_Master0
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
