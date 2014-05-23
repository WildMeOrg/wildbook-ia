#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from os.path import dirname, realpath
"""
PREREQS:
    git
"""

CODE_DIR = dirname(dirname(realpath(__file__)))   # '~/code'
print('__IBEIS_SUPER_SETUP__: %r' % CODE_DIR)

# HACK IN A WAY TO ENSURE UTOOL
try:
    import utool
except ImportError:
    print('FATAL ERROR: UTOOL IS NEEDED FOR SUPER_SETUP')
    import os
    os.chdir(os.path.expanduser(CODE_DIR))
    os.system('git clone https://github.com/Erotemic/utool.git')
    os.chdir('utool')
    os.chdir('python setup.py develop')

#
from utool._internal.meta_util_git import repo_list, set_userid
utool.repo_list = repo_list


def userid_prompt():
    # TODO: Make this prompt for the userid
    return 'Erotemic'

set_userid(userid_prompt())


TPL_REPO_URLS = []
# Test to see if opencv and pyflann have been built
try:
    import cv2
    print('found cv2=%r' % (cv2,))
except ImportError:
    print('need to build opencv')
    TPL_REPO_URLS.append('https://github.com/Erotemic/opencv.git')
try:
    import pyflann  # NOQA
    print('found pyflann=%r' % (pyflann,))
except ImportError:
    print('need to build pyflann')
    TPL_REPO_URLS.append('https://github.com/Erotemic/flann.git')

try:
    import PyQt4
    print('found PyQt4=%r' % (PyQt4,))
except ImportError:
    print('need to install PyQt4')
    raise


(TPL_REPO_URLS,
 TPL_REPO_DIRS) = utool.repo_list(TPL_REPO_URLS, CODE_DIR)


# Non local project repos
(IBEIS_REPO_URLS,
 IBEIS_REPO_DIRS) = utool.repo_list([
     'https://github.com/Erotemic/utool.git',
     'https://github.com/Erotemic/guitool.git',
     'https://github.com/Erotemic/plottool.git',
     'https://github.com/Erotemic/vtool.git',
     'https://github.com/bluemellophone/detecttools.git',
     'https://github.com/Erotemic/hesaff.git',
     'https://github.com/bluemellophone/pyrf.git',
     'https://github.com/Erotemic/ibeis.git',
 ], CODE_DIR)


PROJECT_REPO_URLS = IBEIS_REPO_URLS + TPL_REPO_URLS
PROJECT_REPO_DIRS = IBEIS_REPO_DIRS + TPL_REPO_DIRS

utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)

utool.gg_command('ensure')

if utool.get_flag('--status'):
    utool.gg_command('git status')
    utool.sys.exit(0)

if utool.get_flag('--pull'):
    utool.gg_command('git pull')


if utool.get_flag('--build'):
    # Build tpl repos
    for repo in TPL_REPO_DIRS:
        utool.util_git.std_build_command(repo)
    # Build only IBEIS repos with setup.py
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('python setup.py build')

if utool.get_flag('--develop'):
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('python setup.py develop')


if utool.get_flag('--test'):
    import ibeis
    print('found ibeis=%r' % (ibeis,))
