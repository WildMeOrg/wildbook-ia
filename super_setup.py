#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import dirname, realpath
import sys
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
python_version = platform.python_version()
assert python_version.startswith('2.7'), 'IBEIS is currently limited to python 2.7,  Attempted to run with python %r' % python_version


#pythoncmd = 'python'
#if DISTRO == 'centos':
if sys.platform.startswith('win32'):
    pythoncmd = 'python'
else:
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


def syscmd(cmdstr):
    print('RUN> ' + cmdstr)
    os.system(cmdstr)

# HACK IN A WAY TO ENSURE UTOOL
try:
    print('Checking utool')
    import utool
    utool.set_userid(**userid_prompt())
except Exception:
    print('FATAL ERROR: UTOOL IS NEEDED FOR SUPER_SETUP. Attempting to get utool')
    import os
    import sys
    os.chdir(os.path.expanduser(CODE_DIR))
    print('cloning utool')
    if not os.path.exists('utool'):
        syscmd('git clone https://github.com/Erotemic/utool.git')
    os.chdir('utool')
    print('pulling utool')
    syscmd('git pull')
    print('installing utool for development')
    syscmd('sudo {pythoncmd} setup.py develop'.format(**envcmds))
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
    (('PyQt5', 'PyQt4'),   None)
]

TPL_REPO_URLS = []
# Test to see if opencv and pyflann have been built
for name, repo_url in TPL_MODULES_AND_REPOS:
    try:
        if isinstance(name, str):
            module = __import__(name, globals(), locals(), fromlist=[], level=0)
        else:
            # Allow for multiple module aliases
            module = None
            for name_ in name:
                try:
                    module = __import__(name_, globals(), locals(), fromlist=[], level=0)
                except ImportError as ex:
                    pass
            if module is None:
                raise ex
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
    'https://github.com/Erotemic/ibeis.git',
    #'https://github.com/hjweide/pygist',
], CODE_DIR, forcessh=False)


PROJECT_REPO_URLS = IBEIS_REPO_URLS + TPL_REPO_URLS
PROJECT_REPO_DIRS = IBEIS_REPO_DIRS + TPL_REPO_DIRS

utool.set_project_repos(PROJECT_REPO_URLS, PROJECT_REPO_DIRS)

if utool.get_flag('--status'):
    utool.gg_command('git status')
    utool.sys.exit(0)

if utool.get_flag('--branch'):
    utool.gg_command('git branch')
    utool.sys.exit(0)

utool.gg_command('ensure')

if utool.get_flag('--pull'):
    utool.gg_command('git pull')


if utool.get_flag('--tag-status'):
    utool.gg_command('git tag')

# Tag everything
tag_name = utool.get_arg('--newtag', type_=str, default=None)
if tag_name is not None:
    utool.gg_command('git tag -a "{tag_name}" -m "super_setup autotag {tag_name}"'.format(**locals()))
    utool.gg_command('git push --tags')

if utool.get_flag('--bext'):
    utool.gg_command('{pythoncmd} setup.py build_ext --inplace'.format(**envcmds))

if utool.get_flag('--build'):
    # Build tpl repos
    for repo in TPL_REPO_DIRS:
        utool.util_git.std_build_command(repo)  # Executes {plat}_build.{ext}
    # Build only IBEIS repos with setup.py
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('sudo {pythoncmd} setup.py build'.format(**envcmds))

if utool.get_flag('--develop'):
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('sudo {pythoncmd} setup.py develop'.format(**envcmds))

if utool.get_flag('--test'):
    import ibeis
    print('found ibeis=%r' % (ibeis,))

if utool.get_flag('--push'):
    utool.gg_command('git push')


commit_msg = utool.get_arg('--commit', type_=str, default=None)
if commit_msg is not None:
    utool.gg_command('git commit -am "{commit_msg}"'.format(**locals()))

if utool.get_flag('--clean'):
    utool.gg_command('{pythoncmd} setup.py clean'.format(**envcmds))

# Change Branch
branch_name = utool.get_arg('--checkout', type_=str, default=None)
if branch_name is not None:
    utool.gg_command('git checkout "{branch_name}"'.format(**locals()))

# Creates new branches
newbranch_name = utool.get_arg('--newbranch', type_=str, default=None)
if newbranch_name is not None:
    utool.gg_command('git stash"'.format(**locals()))
    utool.gg_command('git checkout -b "{newbranch_name}"'.format(**locals()))
    utool.gg_command('git stash pop"'.format(**locals()))

# Creates new branches
mergebranch_name = utool.get_arg('--merge', type_=str, default=None)
if mergebranch_name is not None:
    utool.gg_command('git merge "{mergebranch_name}"'.format(**locals()))

newbranch_name2 = utool.get_arg('--newbranch2', type_=str, default=None)
if newbranch_name2 is not None:
    utool.gg_command('git checkout -b "{newbranch_name2}"'.format(**locals()))
    utool.gg_command('git push --set-upstream origin {newbranch_name2}'.format(**locals()))

if utool.get_flag('--serverchmod'):
    utool.gg_command('chmod -R 755 *')

upstream_branch = utool.get_arg('--set-upstream', type_=str, default=None)
if upstream_branch is not None:
    # git 2.0
    utool.gg_command('git branch --set-upstream-to=origin/{upstream_branch} {upstream_branch}'.format(**locals()))


gg_cmd = utool.get_arg('--gg', None)  # global command
if gg_cmd is not None:
    ans = raw_input('Are you sure you want to run: %r on all directories? ' % (gg_cmd,))
    if ans == 'yes':
        utool.gg_command(gg_cmd)
