#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import os
from os.path import join
from util_git import cd
import util_git

PULL = '--pull' in sys.argv

BUNDLE_DPATH = util_git.BUNDLE_DPATH
VIM_REPOS_WITH_SUBMODULES = util_git.VIM_REPOS_WITH_SUBMODULES
VIM_REPO_URLS = util_git.VIM_REPO_URLS
VIM_REPO_DIRS = util_git.get_repo_dirs(VIM_REPO_URLS, BUNDLE_DPATH)
# All modules in the bundle dir (even if not listed)
BUNDLE_DIRS = [join(BUNDLE_DPATH, name) for name in os.listdir(BUNDLE_DPATH)]

cd(BUNDLE_DPATH)

util_git.checkout_repos(VIM_REPO_URLS, VIM_REPO_DIRS)

__NOT_GIT_REPOS__ = []
__BUNDLE_REPOS__ = []


for repodir in BUNDLE_DIRS:
    # Mark which repos do not have .git dirs
    if not util_git.is_gitrepo(repodir):
        __NOT_GIT_REPOS__.append(repodir)
    else:
        __BUNDLE_REPOS__.append(repodir)

if PULL:
    util_git.pull_repos(__BUNDLE_REPOS__, VIM_REPOS_WITH_SUBMODULES)


# Print suggestions for removing nonbundle repos
if len(__NOT_GIT_REPOS__) > 0:
    print('Please fix these nongit repos: ')
    print('\n'.join(__NOT_GIT_REPOS__))
    print('maybe like this: ')
    clutterdir = util_git.unixpath('~/local/vim/vimfiles/clutter')
    suggested_cmds = ['mkdir ' + clutterdir] + [
        'mv ' + util_git.unixpath(dir_) + ' ' + clutterdir for dir_ in __NOT_GIT_REPOS__
    ]
    print('\n'.join(suggested_cmds))
