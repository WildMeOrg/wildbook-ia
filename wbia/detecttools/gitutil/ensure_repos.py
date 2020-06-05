#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import util_git
import __REPOS__


PULL = '--pull' in sys.argv
DEVELOP = '--develop' in sys.argv
CHECK = '--nocheck' not in sys.argv


# Get IBEIS git repository URLS and their local path
wbia_repo_urls = __REPOS__.CODE_REPO_URLS
wbia_repo_dirs = __REPOS__.CODE_REPOS


def checkout_wbia_repos():
    """ Checkout IBEIS repos out if they don't exist """
    util_git.checkout_repos(wbia_repo_urls, wbia_repo_dirs)


def pull_wbia_repos():
    """ Pull IBEIS repos """
    util_git.pull_repos(wbia_repo_dirs)


def setup_develop_wbia_repos():
    """ Install with setuptools using the develop flag """
    util_git.setup_develop_repos(wbia_repo_dirs)


if __name__ == '__main__':
    if CHECK:
        checkout_wbia_repos()
    if PULL:
        pull_wbia_repos()
    if DEVELOP:
        setup_develop_wbia_repos()
