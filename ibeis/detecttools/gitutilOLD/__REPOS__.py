from __future__ import absolute_import, division, print_function
from os.path import expanduser
from itertools import izip
from os.path import normpath, realpath
import platform


def truepath(path):
    return normpath(realpath(expanduser(path)))


def unixpath(path):
    return truepath(path).replace('\\', '/')


# USER DEFINITIONS
CODE_DIR = unixpath('~/code')
BUNDLE_DPATH = unixpath('~/local/vim/vimfiles/bundle')


# Local project repositories
PROJECT_REPOS = map(unixpath, [
    '~/code/detecttools',
    '~/code/flann',
    '~/code/guitool',
    '~/code/hesaff',
    '~/code/ibeis',
    '~/code/IBEIS2014',
    '~/code/plottool',
    '~/code/pyrf',
    '~/code/svm-hog',
    '~/code/templates',
    '~/code/utool',
    '~/code/vtool',
])

# Non local project repos
IBEIS_REPOS_URLS = [
    'https://github.com/Erotemic/utool.git',
    'https://github.com/Erotemic/guitool.git',
    'https://github.com/Erotemic/plottool.git',
    'https://github.com/Erotemic/vtool.git',
    'https://github.com/Erotemic/hesaff.git',
    'https://github.com/Erotemic/ibeis.git',
    'https://github.com/bluemellophone/pyrf.git',
]


def fix_repo_url(repo_url, in_type='https', out_type='ssh'):
    """ Changes the repo_url format """
    format_dict = {
        'https': ('.com/', 'https://'),
        'ssh':   ('.com:', 'git@'),
    }
    for old, new in izip(format_dict[in_type], format_dict[out_type]):
        repo_url = repo_url.replace(old, new)
        return repo_url


def get_computer_name():
    return platform.node()

COMPUTER_NAME  = get_computer_name()

# Check to see if you are on one of Jons Computers
#
IS_OWNER = COMPUTER_NAME in ['Jasons-MacBook-Pro.local',]
if IS_OWNER:
    IBEIS_REPOS_URLS = [fix_repo_url(repo, 'https', 'ssh')
                         for repo in IBEIS_REPOS_URLS]
