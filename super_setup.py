#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from os.path import dirname, realpath
"""
PREREQ:
git config --global push.default current
export CODE_DIR=~/code
mkdir $CODE_DIR
cd $CODE_DIR
git clone https://github.com/Erotemic/ibeis.git
cd ibeis
./_scripts/bootstrap.py
./_scripts/__install_prereqs__.sh
./super_setup.py --build --develop
./super_setup.py --build --develop
"""

print('[super_setup] __IBEIS_SUPER_SETUP__')
CODE_DIR = dirname(dirname(realpath(__file__)))   # '~/code'
print('[super_setup] code_dir: %r' % CODE_DIR)

import platform
(DISTRO, DISTRO_VERSION, DISTRO_TAG) = platform.dist()

pythoncmd = 'python'
if DISTRO == 'centos':
    pythoncmd = 'python2.7'

envcmds = {
    'pythoncmd': pythoncmd
}


def userid_prompt():
    # TODO: Make this prompt for the userid
    if False:
        return {'userid': 'Erotemic', 'permitted_repos': ['pyrf', 'detecttools']}
    else:
        return {}

#################
## ENSURING UTOOL
#################

# HACK IN A WAY TO ENSURE UTOOL
try:
    print('Checking utool')
    import utool
    utool.set_userid(**userid_prompt())
except Exception:
    print('FATAL ERROR: UTOOL IS NEEDED FOR SUPER_SETUP')
    import os
    import sys
    os.chdir(os.path.expanduser(CODE_DIR))
    os.system('git clone https://github.com/Erotemic/utool.git')
    os.chdir('utool')
    os.system('git pull')
    os.system('sudo {pythoncmd} setup.py develop'.format(**envcmds))
    sys.path.append(os.path.realpath(os.getcwd()))
    print('Please rerun super_setup.py')
    sys.exit(1)

utool.init_catch_ctrl_c()

#-----------
# Set userid
#-----------


#-----------
# Third-Party-Libraries
#-----------

print('[super_setup] Checking third-party-libraries')

TPL_MODULES_AND_REPOS = [
    ('cv2',     'https://github.com/Erotemic/opencv.git'),
    ('pyflann', 'https://github.com/Erotemic/flann.git'),
    ('PyQt4',   None)
]

TPL_REPO_URLS = []
# Test to see if opencv and pyflann have been built
for name, repo_url in TPL_MODULES_AND_REPOS:
    try:
        module = __import__(name, globals(), locals(), fromlist=[], level=0)
        print('found %s=%r' % (name, module,))
    except ImportError:
        if repo_url is None:
            raise AssertionError('FATAL ERROR: Need to manually install %s' % name)
        print('need to build %s=%r' % (name, repo_url,))
        TPL_REPO_URLS.append(repo_url)


(TPL_REPO_URLS, TPL_REPO_DIRS) = utool.repo_list(TPL_REPO_URLS, CODE_DIR)

#-----------
# IBEIS project repos
#-----------

# Non local project repos
(IBEIS_REPO_URLS, IBEIS_REPO_DIRS) = utool.repo_list([
    'https://github.com/Erotemic/utool.git',
    'https://github.com/Erotemic/guitool.git',
    'https://github.com/Erotemic/plottool.git',
    'https://github.com/Erotemic/vtool.git',
    'https://github.com/bluemellophone/detecttools.git',
    'https://github.com/Erotemic/hesaff.git',
    'https://github.com/bluemellophone/pyrf.git',
    'https://github.com/hjweide/pygist',
    'https://github.com/Erotemic/ibeis.git',
], CODE_DIR, forcessh=False)


PROJECT_REPO_URLS = IBEIS_REPO_URLS + TPL_REPO_URLS
PROJECT_REPO_DIRS = IBEIS_REPO_DIRS + TPL_REPO_DIRS

utool.set_project_repos(PROJECT_REPO_URLS, PROJECT_REPO_DIRS)

if utool.get_flag('--status'):
    utool.gg_command('git status')
    utool.sys.exit(0)
else:
    utool.gg_command('ensure')

if utool.get_flag('--pull'):
    utool.gg_command('git pull')


if utool.get_flag('--build'):
    # Build tpl repos
    for repo in TPL_REPO_DIRS:
        utool.util_git.std_build_command(repo)
    # Build only IBEIS repos with setup.py
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('sudo {pythoncmd} setup.py build'.format(**envcmds))

if utool.get_flag('--develop'):
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('sudo {pythoncmd} setup.py develop'.format(**envcmds))


if utool.get_flag('--test'):
    import ibeis
    print('found ibeis=%r' % (ibeis,))
