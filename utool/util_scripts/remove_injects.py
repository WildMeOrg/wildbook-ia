#flake8:noqa
"""
This file is broken
"""
from __future__ import absolute_import, division, print_function
from utool import regex


regex_on = r'\(^ \)\([^ ]*rrr.*\)'
repl_off = r'\1pass  # UTOOL_COMMENT \2'


regex_off = r'\(^ \)\(# UTOOL_COMMENT\)\([^ ]*rrr.*\)'
repl_on = r'\1\3'
