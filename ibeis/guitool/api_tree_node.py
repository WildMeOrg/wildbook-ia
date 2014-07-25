from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[tree_node]', DEBUG=False)

try:
    from api_tree_node_cython import *  # NOQA
except ImportError:
    print('[Using Python TreeNode.py')
