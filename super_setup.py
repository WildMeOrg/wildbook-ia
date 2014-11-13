#!/usr/bin/env python2.7
"""
Super Setup
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

# If on current branch
python -c "import utool; utool.copy('super_setup.py', '_ibeis_setup.py')"

# Status
python _ibeis_setup.py -y --gg "git status"
python _ibeis_setup.py -y --gg "git branch"

# Setup Next
#python _ibeis_setup.py -y --gg "git pull"
#python _ibeis_setup.py -y --gg "git checkout master"
#python _ibeis_setup.py -y --gg "git pull"
#python _ibeis_setup.py -y --gg "git checkout -b next"
#python _ibeis_setup.py -y --gg "git checkout next"
#python _ibeis_setup.py -y --gg "git push -u origin next"
#python _ibeis_setup.py -y --gg "git push remote origin/next"
####python _ibeis_setup.py -y --gg "git merge master"


#python _ibeis_setup.py -y --gg "git checkout ^HEAD"
#python _ibeis_setup.py -y --gg "git checkout master"
#python _ibeis_setup.py -y --gg "git checkout pyqt5"


# -- MERGE topic -> next
##python _ibeis_setup.py -y --gg "git checkout pyqt5"
##python _ibeis_setup.py -y --gg "git checkout next"
##python _ibeis_setup.py -y --gg "git merge pyqt5"


# -- MERGE next -> master
python _ibeis_setup.py -y --gg "git checkout next"
python _ibeis_setup.py -y --gg "git merge next"

# Push
python _ibeis_setup.py -y --gg "git push"

#python _ibeis_setup.py -y --gg "git checkout master"
#python _ibeis_setup.py -y --gg "git checkout pyqt5"

GitReferences:
    http://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging
"""
# FUTURE
from __future__ import absolute_import, division, print_function
#-----------------
# SYSTEM ENTRY POINT
# NO UTOOL
# PURE PYTHON
#-----------------
from os.path import dirname, realpath
import platform
import sys
import os

#-----------------
#  UTOOL PYTHON
#-----------------

print('[super_setup] __IBEIS_SUPER_SETUP__')
CODE_DIR = dirname(dirname(realpath(__file__)))   # Home is where the .. is.  # '~/code'
print('[super_setup] code_dir: %r' % CODE_DIR)
(DISTRO, DISTRO_VERSION, DISTRO_TAG) = platform.dist()
python_version = platform.python_version()

# We only support python 2.7
assert '--py3' in sys.argv or python_version.startswith('2.7'), \
    'IBEIS currently needs python 2.7,  Instead got python=%r' % python_version

# Default to python 2.7. Windows is werid
pythoncmd = 'python' if sys.platform.startswith('win32') else 'python2.7'


# TODO: Make this prompt for the userid
def userid_prompt():
    if False:
        return {'userid': 'Erotemic', 'permitted_repos': ['pyrf', 'detecttools']}
    return {}


#################
## ENSURING UTOOL
#################


def syscmd(cmdstr):
    print('RUN> ' + cmdstr)
    os.system(cmdstr)

try:
    # HACK IN A WAY TO ENSURE UTOOL
    print('Checking utool')
    import utool
    utool.set_userid(**userid_prompt())  # FIXME
except Exception:
    #UTOOL_BRANCH = ' -b <branch> <remote_repo>'
    UTOOL_BRANCH = ' -b pyqt5'
    UTOOL_REPO = 'git clone https://github.com/Erotemic/utool.git'
    print('FATAL ERROR: UTOOL IS NEEDED FOR SUPER_SETUP. Attempting to get utool')
    os.chdir(os.path.expanduser(CODE_DIR))
    print('cloning utool')
    if not os.path.exists('utool'):
        syscmd('git clone ' + UTOOL_REPO + '-b' + UTOOL_BRANCH)
    os.chdir('utool')
    print('pulling utool')
    syscmd('git pull')
    print('installing utool for development')
    syscmd('sudo {pythoncmd} setup.py develop'.format(**locals()))
    cwdpath = os.path.realpath(os.getcwd())
    sys.path.append(cwdpath)
    print('Please rerun super_setup.py')
    print(' '.join(sys.argv))
    sys.exit(1)

#-----------------
#  UTOOL PYTHON
#-----------------

utool.init_catch_ctrl_c()

#-----------
# Third-Party-Libraries
#-----------

print('[super_setup] Checking third-party-libraries')

TPL_MODULES_AND_REPOS = [
    ('cv2',     'https://github.com/Erotemic/opencv.git'),
    ('pyflann', 'https://github.com/Erotemic/flann.git'),
    #('yael',    'https://github.com/Erotemic/yael.git'),
    (('PyQt5', 'PyQt4'),   None)
]

TPL_REPO_URLS = []
# Test to see if opencv and pyflann have been built
for nametup, repo_url in TPL_MODULES_AND_REPOS:
    try:
        # Allow for multiple module aliases
        if isinstance(nametup, str):
            nametup = [nametup]
        module = None
        for name_ in nametup:
            try:
                module = __import__(name_, globals(), locals(), fromlist=[], level=0)
            except ImportError as ex:
                pass
        if module is None:
            raise ex
        print('found %s=%r' % (nametup, module,))
    except ImportError:
        assert repo_url is not None, ('FATAL ERROR: Need to manually install %s' % nametup)
        print('!!! NEED TO BUILD %s=%r' % (nametup, repo_url,))
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
    'https://github.com/aweinstock314/cyth.git',
    #'https://github.com/hjweide/pygist',
], CODE_DIR, forcessh=False)


PROJECT_REPO_URLS = IBEIS_REPO_URLS + TPL_REPO_URLS
PROJECT_REPO_DIRS = IBEIS_REPO_DIRS + TPL_REPO_DIRS

# Set utool global git repos
utool.set_project_repos(PROJECT_REPO_URLS, PROJECT_REPO_DIRS)

GET_ARGFLAG = utool.get_flag
GET_ARGVAL = utool.get_arg

# Commands on global git repos
if GET_ARGFLAG('--status'):
    utool.gg_command('git status')
    utool.sys.exit(0)

if GET_ARGFLAG('--branch'):
    utool.gg_command('git branch')
    utool.sys.exit(0)

utool.gg_command('ensure')

if GET_ARGFLAG('--pull'):
    utool.gg_command('git pull')


if GET_ARGFLAG('--tag-status'):
    utool.gg_command('git tag')

# Tag everything
tag_name = GET_ARGVAL('--newtag', type_=str, default=None)
if tag_name is not None:
    utool.gg_command('git tag -a "{tag_name}" -m "super_setup autotag {tag_name}"'.format(**locals()))
    utool.gg_command('git push --tags')

if GET_ARGFLAG('--bext'):
    utool.gg_command('{pythoncmd} setup.py build_ext --inplace'.format(**locals()))

if GET_ARGFLAG('--build'):
    # Build tpl repos
    for repo in TPL_REPO_DIRS:
        utool.util_git.std_build_command(repo)  # Executes {plat}_build.{ext}
    # Build only IBEIS repos with setup.py
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('sudo {pythoncmd} setup.py build'.format(**locals()))

if GET_ARGFLAG('--develop'):
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('sudo {pythoncmd} setup.py develop'.format(**locals()))

if GET_ARGFLAG('--install'):
    utool.set_project_repos(IBEIS_REPO_URLS, IBEIS_REPO_DIRS)
    utool.gg_command('python setup.py install'.format(**locals()))

if GET_ARGFLAG('--test'):
    import ibeis
    print('found ibeis=%r' % (ibeis,))

if GET_ARGFLAG('--push'):
    utool.gg_command('git push')


commit_msg = GET_ARGVAL('--commit', type_=str, default=None)
if commit_msg is not None:
    utool.gg_command('git commit -am "{commit_msg}"'.format(**locals()))

if GET_ARGFLAG('--clean'):
    utool.gg_command('{pythoncmd} setup.py clean'.format(**locals()))

# Change Branch
branch_name = GET_ARGVAL('--checkout', type_=str, default=None)
if branch_name is not None:
    utool.gg_command('git checkout "{branch_name}"'.format(**locals()))

# Creates new branches
newbranch_name = GET_ARGVAL('--newbranch', type_=str, default=None)
if newbranch_name is not None:
    utool.gg_command('git stash"'.format(**locals()))
    utool.gg_command('git checkout -b "{newbranch_name}"'.format(**locals()))
    utool.gg_command('git stash pop"'.format(**locals()))

# Creates new branches
mergebranch_name = GET_ARGVAL('--merge', type_=str, default=None)
if mergebranch_name is not None:
    utool.gg_command('git merge "{mergebranch_name}"'.format(**locals()))

newbranch_name2 = GET_ARGVAL('--newbranch2', type_=str, default=None)
if newbranch_name2 is not None:
    utool.gg_command('git checkout -b "{newbranch_name2}"'.format(**locals()))
    utool.gg_command('git push --set-upstream origin {newbranch_name2}'.format(**locals()))

if GET_ARGFLAG('--serverchmod'):
    utool.gg_command('chmod -R 755 *')

if GET_ARGFLAG('--chown'):
    username = os.environ['USERNAME']
    usergroup = username
    utool.gg_command('sudo chown -R {username}:{usergroup} *'.format(**locals()))

upstream_branch = GET_ARGVAL('--set-upstream', type_=str, default=None)
if upstream_branch is not None:
    # git 2.0
    utool.gg_command('git branch --set-upstream-to=origin/{upstream_branch} {upstream_branch}'.format(**locals()))


upstream_push = GET_ARGVAL('--upstream-push', type_=str, default=None)
if upstream_push is not None:
    utool.gg_command('git push --set-upstream origin {upstream_push}'.format(**locals()))


gg_cmd = GET_ARGVAL('--gg', None)  # global command
if gg_cmd is not None:
    ans = 'yes' if GET_ARGFLAG('-y') else raw_input('Are you sure you want to run: %r on all directories? ' % (gg_cmd,))
    if ans == 'yes':
        utool.gg_command(gg_cmd)
