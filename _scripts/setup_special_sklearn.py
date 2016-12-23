#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python ~/code/ibeis/_scripts/setup_special_sklearn.py
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import utool as ut
# import git
# import os

target = 'dev_combo'
master = 'master'

mixins = [
    # 'mbkm_fixup',
    'progiter',
    'multiclass_mcc',
    'missing_values_rf',
]

# dpath = os.getcwd()
dpath = ut.truepath('~/code/scikit-learn')
utrepo = ut.Repo(dpath=dpath)

gitrepo = utrepo.as_gitpython()

# Checkout master, ensure its up to date, and recreate the combo branch
utrepo.issue('git checkout ' + master)

if True:
    utrepo.issue('git pull')
    utrepo.issue('git fetch --all')

    for branch in mixins:
        utrepo.issue('git checkout ' + branch)
        # utrepo.reset_branch_to_remote(branch)
        utrepo.issue('git pull')

if target in utrepo.branches:
    utrepo.issue('git branch -D ' + target)
utrepo.issue('git checkout -b ' + target)

# # Attempt to automerge taking whatever is in the mixin branches as de-facto
for branch in mixins:
    # try:
    #     utrepo.issue('git merge --no-edit -s recursive '  + branch)
    # except Exception:
    # utrepo.issue('git merge --no-edit -s recursive -Xours ' + branch)
    utrepo.issue('git merge --no-edit -s recursive -Xtheirs ' + branch)
    # break
    # utrepo.issue('git commit -am "merge"')
    # utrepo.issue('git merge --no-edit ' + branch)

# git merge


# # Recompile the
utrepo.issue('python setup.py clean')
utrepo.issue('python setup.py develop')
# # utrepo.reset_branch_to_remote('speedup_kmpp')
