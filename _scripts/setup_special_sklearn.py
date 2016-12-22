#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import utool as ut
# import git
# import os

target = 'dev_combo'
base = 'master'

mixins = [
    'multiclass_mcc',
    'missing_values_rf',
    'mbkm_fixup',
    'progiter',
]

# dpath = os.getcwd()
dpath = ut.truepath('~/code/sklearn')
utrepo = ut.Repo(dpath=dpath)

gitrepo = utrepo.as_gitpython()

# Checkout master, ensure its up to date, and recreate the combo branch
utrepo.issue('git checkout ' + base)
utrepo.issue('git pull')
if target in utrepo.branches:
    utrepo.issue('git branch -D ' + target)
utrepo.issue('git checkout -b ' + target)

# Attempt to automerge taking whatever is in the mixin branches as de-facto
for branch in mixins:
    utrepo.issue('git merge --no-edit --strategy=ours ' + branch)


# Recompile the
utrepo.issue('python setup clean')
utrepo.issue('python setup develop')
# utrepo.reset_branch_to_remote('speedup_kmpp')
