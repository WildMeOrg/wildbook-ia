"""
This will be a wrapper around pyflann.

Splits kd-forests into multiple kd-forests to alleviate build time for large
databases.
"""
from __future__ import absolute_import, division, print_function
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[stack_tree]', DEBUG=False)


class StackTree(object):
    def __init__(self):
        self.children = []  # StackTrees
