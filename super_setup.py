#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
TODO:
    * needs to check if required modules are installed (or prefereably developed)
    * needs to be able to ignore plugins that the user doesnt care about

Super Setup
PREREQ:
git config --global push.default current
export CODE_DIR=~/code
mkdir $CODE_DIR
cd $CODE_DIR
git clone https://github.com/Erotemic/ibeis.git
cd ibeis

python super_setup.py --bootstrap
OR
./_scripts/bootstrap.py
THEN
./_scripts/__install_prereqs__.sh
THEN
./super_setup.py --build --develop
./super_setup.py --build --develop

./super_setup.py --status

# If on current branch copy so super setup isn't overwriten as we go
python -c "import utool as ut; ut.copy('super_setup.py', '_ibeis_setup.py')"

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
#python _ibeis_setup.py -y --gg "git checkout next"


# -- MERGE topic -> next
##python _ibeis_setup.py -y --gg "git checkout topic"
##python _ibeis_setup.py -y --gg "git checkout next"
##python _ibeis_setup.py -y --gg "git merge topic"


# -- MERGE next -> master
python _ibeis_setup.py -y --gg "git checkout master"
python _ibeis_setup.py -y --gg "git merge next"

# -- SAFER MERGE topic -> next
python super_setup.py --checkout next
python super_setup.py --newlocalbranch merge_next_joncrall_dev_branch
python super_setup.py --merge joncrall_dev_branch
./run_tests.py
python super_setup.py --checkout next
python super_setup.py --merge merge_next_joncrall_dev_branch

# Push
python _ibeis_setup.py -y --gg "git push"

#python _ibeis_setup.py -y --gg "git checkout master"
#python _ibeis_setup.py -y --gg "git checkout next"


# MAKE A NEW BRANCH
python super_setup.py --newbranch joncrall_dev_branch
python super_setup.py --checkout joncrall_dev_branch
python super_setup.py --checkout next

python super_setup.py --newbranch jdb
python super_setup.py --checkout jdb


GitReferences:
    http://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging

FIXME:
    graph-viz
    pydot
    ibeis_cnn
    Theano
    Lasange
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import dirname, realpath
import platform
import sys
import os

#-----------------
# SYSTEM ENTRY POINT, NO UTOOL, BARE PYTHON
#-----------------


USAGE = '''

 --- USAGE ---

IBEIS (IMAGE ANALYSIS) SUPER SETUP

This script is meant to help setup, install, and update the developer
enviroment for IBEIS Image Analysis.

****
# Step 1 Initial Development Prereqs:

The first thing is to ensure you have a baseline development enviroment (gcc,
fortran, cmake, blas, git, pip, etc...).  This should work well for apt-get,
yum, and macports package managers.  It is possible to use Windows MinGW, but
it is not well supported.

The following command outputs the commands to install these prereq packages.

    python super_setup.py --bootstrap

****
# Step 2 - utool

Just running the script will download and install utool --- a utility library
used in all aspects of the system.

    python super_setup.py

****
# Step 3 - Download / Update Image Analysis Packages

Running the script again once utool is installed will ensure the rest of the
repositories are cloned and on your machine in the directory above this one, or
in a custom location set by your $CODE_DIR environment variable.  Running with
the pull command will update the packages as well.

    python super_setup.py pull

****
# Step 4 - Build C++ compoments.

Some submodles require C++ libraries. Build them using the following Command.

    python super_setup.py build

****
# Step 5 - Install the system.

Register these packages with the python enviroment.

    python super_setup.py develop

 --- /USAGE ---
'''

print('''
IBEIS Image Analysis (IA)
____ _  _ ___  ____ ____    ____ ____ ___ _  _ ___
[__  |  | |__] |___ |__/    [__  |___  |  |  | |__]
___] |__| |    |___ |  \    ___] |___  |  |__| |
''')

show_usage = len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']
if show_usage:
    print(USAGE)

print('USER = %r' % os.getenv("USER"))


def is_running_as_root():
    """
    References:
        http://stackoverflow.com/questions/5721529/running-python-script-as-root
        http://stackoverflow.com/questions/2806897/checking-script-has-root
    """
    return os.getenv('USER') == 'root'

if is_running_as_root():
    print('Do not run super_setup.py as root')
    sys.exit(1)

WIN32 = sys.platform.startswith('win32')

print('[super_setup] __IBEIS_SUPER_SETUP__')

if 'CODE_DIR' in os.environ:
    CODE_DIR = os.environ.get('CODE_DIR')
else:
    CODE_DIR = dirname(dirname(realpath(__file__)))   # Home is where the .. is.  # '~/code'

print('[super_setup] code_dir: %r' % CODE_DIR)
(DISTRO, DISTRO_VERSION, DISTRO_TAG) = platform.dist()
python_version = platform.python_version()

# We only support python 2.7
PY3 = '--py3' in sys.argv
assert PY3  or python_version.startswith('2.7'), \
    'IBEIS currently supports python 2.7,  Instead got python=%r. use --py3 to override' % python_version

# Default to python 2.7. Windows is werid

if PY3:
    pythoncmd = 'python3'
else:
    pythoncmd = 'python' if WIN32 else 'python2.7'


def bootstrap():
    def import_module_from_fpath(module_fpath):
        """ imports module from a file path """
        import platform
        from os.path import basename, splitext
        python_version = platform.python_version()
        modname = splitext(basename(module_fpath))[0]
        if python_version.startswith('2.7'):
            import imp
            module = imp.load_source(modname, module_fpath)
        elif python_version.startswith('3'):
            import importlib.machinery
            loader = importlib.machinery.SourceFileLoader(modname, module_fpath)
            module = loader.load_module()
        else:
            raise AssertionError('invalid python version')
        return module

    if WIN32:
        # need to preinstall parse
        win32bootstrap_fpath = os.path.abspath('_scripts/win32bootstrap.py')
        win32bootstrap = import_module_from_fpath(win32bootstrap_fpath)
        win32bootstrap.bootstrap_sysreq()

    else:
        #import bootstrap
        bootstrap_fpath = os.path.abspath('_scripts/bootstrap.py')
        bootstrap = import_module_from_fpath(bootstrap_fpath)
        #sys.path.append(os.path.abspath('_scripts'))
        bootstrap.bootstrap_sysreq()
    sys.exit(0)

if '--bootstrap' in sys.argv or 'bootstrap' in sys.argv:
    bootstrap()


#################
#  ENSURING UTOOL
#################


def ensure_utool():
    def syscmd(cmdstr):
        print('RUN> ' + cmdstr)
        os.system(cmdstr)
    #UTOOL_BRANCH = ' -b <branch> <remote_repo>'
    UTOOL_BRANCH = 'next'
    UTOOL_REPO = 'https://github.com/Erotemic/utool.git'
    print('FATAL ERROR: UTOOL IS NEEDED FOR SUPER_SETUP. Attempting to get utool')
    cwdpath = os.path.realpath(os.getcwd())
    usr_code_dir = os.path.expanduser(CODE_DIR)
    os.chdir(usr_code_dir)
    print("user code dir = %r" % usr_code_dir)
    print('cloning utool')
    if not os.path.exists('utool'):
        syscmd('git clone ' + UTOOL_REPO + ' -b ' + UTOOL_BRANCH)
    os.chdir('utool')
    print('pulling utool')
    syscmd('git pull')
    print('installing utool for development')
    cmdstr = '{pythoncmd} setup.py develop'.format(pythoncmd=pythoncmd)
    in_virtual_env = hasattr(sys, 'real_prefix')
    if not WIN32 and not in_virtual_env:
        cmdstr = 'sudo ' + cmdstr
    syscmd(cmdstr)
    os.chdir(cwdpath)
    sys.path.append(usr_code_dir)
    print('Please rerun super_setup.py')
    print(' '.join(sys.argv))
    if '--check-utool-error-code-0' in sys.argv:
        sys.exit(0)
    else:
        sys.exit(1)

try:
    # HACK IN A WAY TO ENSURE UTOOL
    print('Checking utool')
    import utool as ut
except Exception:
    ensure_utool()

#-----------------
#  UTOOL PYTHON
#-----------------


def GET_ARGFLAG(arg, *args, **kwargs):
    return arg.lstrip('--') in sys.argv or ut.get_argflag(arg, *args, **kwargs)

GET_ARGVAL = ut.get_argval

ut.init_catch_ctrl_c()


WITH_CNN = True
WITH_PYRF = True
WITH_TPL = True
WITH_QT = not ut.get_argflag('--no-qt')
WITH_GUI = not ut.get_argflag('--no-gui')
WITH_CUSTOM_TPL = True
WITH_FLUKEMATCH = True

#-----------
# IBEIS project repos
#-----------
ibeis_rman = ut.RepoManager([
    'https://github.com/Erotemic/utool.git',
    'https://github.com/Erotemic/vtool.git',
    'https://github.com/Erotemic/dtool.git',
    'https://github.com/bluemellophone/detecttools.git',
], CODE_DIR, label='core')

tpl_rman = ut.RepoManager([], CODE_DIR, label='tpl')
if not GET_ARGFLAG('--ignore-opencv'):
    tpl_rman.add_repo(ut.Repo('https://github.com/Itseez/opencv.git', CODE_DIR, modname='cv2'))

if WITH_GUI:
    ibeis_rman.add_repos([
        'https://github.com/Erotemic/plottool.git',
    ])

    if WITH_QT:
        ibeis_rman.add_repos([
            'https://github.com/Erotemic/guitool.git',
        ])
        tpl_rman.add_repo(ut.Repo(modname=('PyQt4', 'PyQt5')))

if WITH_CUSTOM_TPL:
    ibeis_rman.add_repo(ut.Repo('https://github.com/Erotemic/flann.git', CODE_DIR, modname='pyflann'))
    ibeis_rman.add_repos([
        'https://github.com/Erotemic/hesaff.git',
    ])

if WITH_CNN:
    ibeis_rman.add_repos([
        'https://github.com/bluemellophone/ibeis_cnn',
        'https://github.com/bluemellophone/pydarknet.git',
    ])
    if WITH_FLUKEMATCH:
        ibeis_rman.add_repos([
            'https://github.com/zmjjmz/ibeis-flukematch-module.git'
        ])
    # CNN Dependencies
    tpl_rman.add_repos([
        'https://github.com/Theano/Theano.git',
        'https://github.com/lisa-lab/pylearn2.git',
        'https://github.com/Erotemic/Lasagne.git',
    ])

if WITH_PYRF:
    ibeis_rman.add_repos([
        'https://github.com/bluemellophone/pyrf.git',
    ])

if False:
    # Depricated
    ibeis_rman.add_repos([
        #'https://github.com/bluemellophone/pybing.git',
        #'https://github.com/aweinstock314/cyth.git',
        #'https://github.com/hjweide/pygist',
    ])

# Add main repo (Must be checked last due to dependency issues)
ibeis_rman.add_repos([
    'https://github.com/Erotemic/ibeis.git',
])


#-----------
# Custom third party build/install scripts
#-----------

def define_custom_scripts(tpl_rman):
    if ut.in_virtual_env():
        fmtdict = {
            'sys_dist_packages': ut.get_global_dist_packages_dir(),
            'venv_site_packages': ut.get_site_packages_dir(),
        }
        # Allows us to use a system qt install in a virtual environment.
        system_to_venv = ut.codeblock(
            r"""
            # STARTBLOCK bash
            ln -s {sys_dist_packages}/PyQt4/ {venv_site_packages}/PyQt4
            ln -s {sys_dist_packages}/sip*.so {venv_site_packages}/
            ln -s {sys_dist_packages}/sip*.py {venv_site_packages}/
            # ENDBLOCK bash
            """).format(**fmtdict)
        tpl_rman['PyQt4'].add_script('system_to_venv', system_to_venv)
        # TODO: add custom build alternative
        pass
    else:
        pass
    """
    # http://stackoverflow.com/questions/18042919/how-to-install-pyqt5-on-a-new-virtualenv-and-work-on-an-idle
    pip install vext.pyqt5
    sudo apt-get install pyqt5-dev
    sudo apt-get install python3-pyqt5
    python
    python -c "import sip; print('[test] Python can import sip')"
    python -c "import sip; print('sip.__file__=%r' % (sip.__file__,))"
    python -c "import sip; print('sip.SIP_VERSION=%r' % (sip.SIP_VERSION,))"
    python -c "import sip; print('sip.SIP_VERSION_STR=%r' % (sip.SIP_VERSION_STR,))"

    ln -s /usr/lib/python3/dist-packages/PyQt5/ /home/joncrall/venv3/lib/python3.4/site-packages/PyQt5
    ln -s /usr/lib/python3/dist-packages/sip*.so /home/joncrall/venv3/lib/python3.4/site-packages/
    ln -s /usr/lib/python3/dist-packages/sip*.py /home/joncrall/venv3/lib/python3.4/site-packages/

    """

    ibeis_rman['pyflann'].add_script('install', ut.codeblock(
        r'''
        # STARTBLOCK bash
        # The pyflann source lives here
        cd {repo_dir}/src/python
        # But the setup script is generated during build
        python {repo_dir}/build/src/python/setup.py develop
        # ENDBLOCK bash
        ''').format(repo_dir=ibeis_rman['pyflann'].dpath)
    )

    if PY3:
        tpl_rman['cv2'].add_script('build', ut.codeblock(
            r"""
            # STARTBLOCK bash
            cd $CODE_DIR
            git clone https://github.com/Itseez/opencv.git
            cd opencv
            # Get Extras
            git clone https://github.com/Itseez/opencv_contrib.git
            mkdir -p build34
            cd build34

            if [[ "$VIRTUAL_ENV" == ""  ]]; then
                export LOCAL_PREFIX=/usr/local
                export PYTHON3_PACKAGES_PATH=$LOCAL_PREFIX/lib/python3.4/dist-packages
                export _SUDO="sudo"
            else
                export LOCAL_PREFIX=$VIRTUAL_ENV/local
                export PYTHON3_PACKAGES_PATH=$LOCAL_PREFIX/lib/python3.4/site-packages
                export _SUDO=""
            fi

            echo "LOCAL_PREFIX = $LOCAL_PREFIX"
            echo "PYTHON3_PACKAGES_PATH = $PYTHON3_PACKAGES_PATH"
            # use dist packages on ubuntu. may need to change for other platforms
            cmake -G "Unix Makefiles" \
                -D WITH_OPENMP=ON \
                -D CMAKE_BUILD_TYPE=RELEASE \
                -D BUILD_opencv_python2=Off \
                -D BUILD_opencv_python3=On \
                -D PYTHON3_PACKAGES_PATH=$PYTHON3_PACKAGES_PATH \
                -D CMAKE_INSTALL_PREFIX=$LOCAL_PREFIX \
                -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
                ..
            export PYTHON_PACKAGES_PATH=$PYTHON3_PACKAGES_PATH

            export NCPUS=$(grep -c ^processor /proc/cpuinfo)
            make -j$NCPUS
            # ENDBLOCK
            """))
    else:
        # TODO: allow system installation as well
        tpl_rman['cv2'].add_script('build', ut.codeblock(
            r"""
            # STARTBLOCK bash
            cd $CODE_DIR
            git clone https://github.com/Itseez/opencv.git
            cd opencv
            # Get Extras
            git clone https://github.com/Itseez/opencv_contrib.git
            mkdir -p build27
            cd build27

            if [[ "$VIRTUAL_ENV" == ""  ]]; then
                export LOCAL_PREFIX=/usr/local
                export PYTHON2_PACKAGES_PATH=$LOCAL_PREFIX/lib/python2.7/dist-packages
                export _SUDO="sudo"
            else
                export LOCAL_PREFIX=$VIRTUAL_ENV/local
                export PYTHON2_PACKAGES_PATH=$LOCAL_PREFIX/lib/python2.7/site-packages
                export _SUDO=""
            fi
            export PYTHON_PACKAGES_PATH=$PYTHON2_PACKAGES_PATH

            echo "LOCAL_PREFIX = $LOCAL_PREFIX"
            echo "PYTHON2_PACKAGES_PATH = $PYTHON2_PACKAGES_PATH"
            # use dist packages on ubuntu. may need to change for other platforms
            cmake -G "Unix Makefiles" \
                -D WITH_OPENMP=ON \
                -D CMAKE_BUILD_TYPE=RELEASE \
                -D PYTHON2_PACKAGES_PATH=$PYTHON2_PACKAGES_PATH \
                -D CMAKE_INSTALL_PREFIX=$LOCAL_PREFIX \
                -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
                ..

            export NCPUS=$(grep -c ^processor /proc/cpuinfo)
            make -j$NCPUS
            # ENDBLOCK
            """))

    tpl_rman['cv2'].add_script('install', ut.codeblock(
        r"""
        # STARTBLOCK bash
        $_SUDO make install
        # Hack because cv2 does not want to be installed for some reason
        cp lib/cv2.so $PYTHON_PACKAGES_PATH
        # Test makesure things working
        python -c "import numpy; print(numpy.__file__)"
        python -c "import numpy; print(numpy.__version__)"
        python -c "import cv2; print(cv2.__version__)"
        python -c "import cv2; print(cv2.__file__)"
        #python -c "import vtool"
        # Check if we have contrib modules
        python -c "import cv2; print(cv2.xfeatures2d)"
        # ENDBLOCK
        """)
    )

#     """
#     export THEANO_FLAGS="device=cpu,print_active_device=True,enable_initial_driver_test=True"
#     set THEANO_FLAGS=device=cpu,print_active_device=True,enable_initial_driver_test=True,print_test_value=True

#     python -c "import pydot; print(pydot.__file__)"
#     python -c "import pydot; print(pydot.__version__)"
#     python -c "import pydot; print(pydot.find_graphviz())"
#     python -c "import theano; print(theano.__file__)"
#     python -c "import pylearn2; print(pylearn2.__file__)"
#     python -c "import lasagne; print(lasagne.__file__)"
#     python -c "import ibeis_cnn; print(ibeis_cnn.__file__)"
#     python -c "import detecttools; print(detecttools.__file__)"
#     """

#-----------
# Verify TPL Dependencies
#-----------

if 1:
    for repo in tpl_rman.repos:
        print('python -c "import {0}; print({0}.__file__)"'.format(repo.modname))
        print('python -c "import {0}; print({0}.__version__)"'.format(repo.modname))


#-----------
# Execute Commands on Core Repos
#-----------

print('ibeis_rman = %r' % (ibeis_rman,))
#import utool
#utool.embed()

# Commands on global git repos
if GET_ARGFLAG('--status'):
    ibeis_rman.issue('git status')
    sys.exit(0)

ibeis_rman.ensure()

if GET_ARGFLAG('--fix') or GET_ARGFLAG('--check'):
    missing_dynlib = tpl_rman.check_cpp_build()
    missing_dynlib += ibeis_rman.check_cpp_build()

    missing_install = tpl_rman.check_installed()
    missing_install += ibeis_rman.check_installed()

    problems = []
    problems += ibeis_rman.check_importable()
    problems += tpl_rman.check_importable()

if GET_ARGFLAG('--fix'):
    print('Trying to fix problems')

    for repo in missing_dynlib:
        repo.custom_build()

    for repo, recommended_fix in problems:
        print('Trying to fix repo = %r' % (repo,))
        print(' * recommended_fix = %r' % (recommended_fix,))
        if recommended_fix == 'rebuild':
            repo.custom_build()
            print('Can currently only fix one module at a time. Please re-run')
            sys.exit(1)
        else:
            print('Not sure how to fix %r' % (repo,))


if GET_ARGFLAG('--pull'):
    ibeis_rman.issue('git pull')

if GET_ARGFLAG('--build'):
    # Build tpl repos
    tpl_rman.custom_build()
    ibeis_rman.custom_build()
    # Build only IBEIS repos with setup.py
    _rman = ibeis_rman.only_with_pysetup()
    _rman.issue('{pythoncmd} setup.py build'.format(**locals()))

# Like install, but better if you are developing
if GET_ARGFLAG('--develop'):
    _rman = ibeis_rman.only_with_pysetup()
    _rman.issue('{pythoncmd} setup.py develop'.format(**locals()),
                sudo=not ut.in_virtual_env())

if GET_ARGFLAG('--clean'):
    _rman = ibeis_rman.only_with_pysetup()
    _rman.issue('{pythoncmd} setup.py clean'.format(**locals()))

if GET_ARGFLAG('--install'):
    print('WARNING: Dont use install if you are a developer. Use develop instead.')
    _rman = ibeis_rman.only_with_pysetup()
    _rman.issue('python setup.py install'.format(**locals()))

if GET_ARGFLAG('--push'):
    ibeis_rman.issue('git push')

if GET_ARGFLAG('--branch'):
    ibeis_rman.issue('git branch')
    sys.exit(0)

if GET_ARGFLAG('--tag-status'):
    ibeis_rman.issue('git tag')

# Tag everything
tag_name = GET_ARGVAL('--newtag', type_=str, default=None)
if tag_name is not None:
    ibeis_rman.issue('git tag -a "{tag_name}" -m "super_setup autotag {tag_name}"'.format(**locals()))
    ibeis_rman.issue('git push --tags')

if GET_ARGFLAG('--bext'):
    ibeis_rman.issue('{pythoncmd} setup.py build_ext --inplace'.format(**locals()))


commit_msg = GET_ARGVAL('--commit', type_=str, default=None)
if commit_msg is not None:
    ibeis_rman.issue('git commit -am "{commit_msg}"'.format(**locals()))

# Change Branch
branch_name = GET_ARGVAL('--checkout', type_=str, default=None)
if branch_name is not None:
    try:
        ibeis_rman.issue('git checkout "{branch_name}"'.format(**locals()))
    except Exception:
        print('ERROR: Could not checkout branch: %r' % (branch_name, ))

# Creates new branches
newbranch_name = GET_ARGVAL('--newbranch', type_=str, default=None)
if newbranch_name is not None:
    #rman.issue('git stash"'.format(**locals()))
    ibeis_rman.issue('git checkout -b "{newbranch_name}"'.format(**locals()))
    ibeis_rman.issue('git push --set-upstream origin {newbranch_name}'.format(**locals()))
    #rman.issue('git stash pop"'.format(**locals()))

# Creates new branches
newlocalbranch_name = GET_ARGVAL('--newlocalbranch', type_=str, default=None)
if newlocalbranch_name is not None:
    #rman.issue('git stash"'.format(**locals()))
    ibeis_rman.issue('git checkout -b "{newlocalbranch_name}"'.format(**locals()))
    #rman.issue('git push --set-upstream origin {newlocalbranch_name}'.format(**locals()))
    #rman.issue('git stash pop"'.format(**locals()))

# Creates new branches
mergebranch_name = GET_ARGVAL('--merge', type_=str, default=None)
if mergebranch_name is not None:
    ibeis_rman.issue('git merge "{mergebranch_name}"'.format(**locals()))

# Change ownership
if GET_ARGFLAG('--serverchmod'):
    ibeis_rman.issue('chmod -R 755 *')

if GET_ARGFLAG('--chown'):
    # Fixes problems where repos are checked out as root
    username = os.environ.get('USERNAME', ut.get_argval('--username'))
    if username is None:
        username = os.environ.get('USER', None)
    if username is None:
        raise AssertionError('cannot find username in commandline or environment vars')
    usergroup = username
    ibeis_rman.issue('chown -R {username}:{usergroup} *'.format(**locals()),
                     sudo=True)

upstream_branch = GET_ARGVAL('--set-upstream', type_=str, default=None)
if upstream_branch is not None:
    # git 2.0
    ibeis_rman.issue('git branch --set-upstream-to=origin/{upstream_branch} {upstream_branch}'.format(**locals()))


upstream_push = GET_ARGVAL('--upstream-push', type_=str, default=None)
if upstream_push is not None:
    ibeis_rman.issue('git push --set-upstream origin {upstream_push}'.format(**locals()))


if GET_ARGFLAG('--test'):
    failures = []
    for repo_dpath in ibeis_rman.repo_dirs:
        # ut.getp_
        mod_dpaths = ut.get_submodules_from_dpath(repo_dpath, recursive=False,
                                                  only_packages=True)
        modname_list = ut.lmap(ut.get_modname_from_modpath, mod_dpaths)
        print('Checking modules = %r' % (modname_list,))

        for modname in modname_list:
            try:
                ut.import_modname(modname)
                print(modname + ' success')
            except ImportError as ex:
                failures += [modname]
                print(modname + ' failure')

    print('failures = %s' % (ut.repr3(failures),))
    # print('repo_dpath = %r' % (repo_dpath,))
    # print('modules = %r' % (modules,))
    # import ibeis
    # print('found ibeis=%r' % (ibeis,))


if False:
    try:
        from six.moves import input
    except ImportError:
        input = raw_input
    # General global git command
    gg_cmd = GET_ARGVAL('--gg', None)  # global command
    if gg_cmd is not None:
        ans = 'yes' if GET_ARGFLAG('-y') else input('Are you sure you want to run: %r on all directories? ' % (gg_cmd,))
        if ans == 'yes':
            ibeis_rman.issue(gg_cmd)
