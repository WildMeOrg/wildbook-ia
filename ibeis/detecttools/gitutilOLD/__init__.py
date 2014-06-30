from __future__ import absolute_import, division, print_function
import sys
import os
from itertools import izip
from os.path import expanduser, normpath, realpath, exists, join, dirname, split, isdir
import detecttools.gitutil.__REPOS__

PROJECT_REPOS    = __REPOS__.PROJECT_REPOS
IBEIS_REPOS_URLS = __REPOS__.IBEIS_REPOS_URLS
CODE_DIR         = __REPOS__.CODE_DIR


def truepath(path):
    return normpath(realpath(expanduser(path)))


def unixpath(path):
    return truepath(path).replace('\\', '/')


def cd(dir_):
    dir_ = truepath(dir_)
    print('> cd ' + dir_)
    os.chdir(dir_)


def cmd(command):
    print('> ' + command)
    os.system(command)


def gitcmd(repo, command):
    print()
    print("************")
    print(repo)
    os.chdir(repo)
    os.system(command)
    print("************")


def gg_command(command):
    """ Runs a command on all of your PROJECT_REPOS """
    for repo in PROJECT_REPOS:
        if exists(repo):
            gitcmd(repo, command)


def get_repo_dname(repo_url):
    """ Break url into a dirname """
    slashpos = repo_url.rfind('/')
    colonpos = repo_url.rfind(':')
    if slashpos != -1 and slashpos > colonpos:
        pos = slashpos
    else:
        pos = colonpos
    repodir = repo_url[pos + 1:].replace('.git', '')
    return repodir


def get_repo_dirs(repo_urls, checkout_dir):
    repo_dirs = [join(checkout_dir, get_repo_dname(url)) for url in repo_urls]
    return repo_dirs


def checkout_repos(repo_urls, repo_dirs=None, checkout_dir=None):
    """ Checkout every repo in repo_urls into checkout_dir """
    # Check out any repo you dont have
    if checkout_dir is not None:
        repo_dirs = get_repo_dirs(checkout_dir)
    assert repo_dirs is not None, 'specify checkout dir or repo_dirs'
    for repodir, repourl in izip(repo_dirs, repo_urls):
        print('[git] checkexist: ' + repodir)
        if not exists(repodir):
            cd(dirname(repodir))
            cmd('git clone ' + repourl)


def setup_develop_repos(repo_dirs):
    """ Run python installs """
    for repodir in repo_dirs:
        print('Installing: ' + repodir)
        cd(repodir)
        assert exists('setup.py'), 'cannot setup a nonpython repo'
        cmd('python setup.py develop')


def status_repos():
    for repodir in PROJECT_REPOS:
        print('\n\nStatus: ' + repodir)
        cd(repodir)
        assert exists('.git'), 'cannot status a nongit repo'
        cmd('git status')

        
def pull_repos(repo_dirs, repos_with_submodules=[]):
    for repodir in repo_dirs:
        print('Pulling: ' + repodir)
        cd(repodir)
        assert exists('.git'), 'cannot pull a nongit repo'
        cmd('git pull')
        reponame = split(repodir)[1]
        if reponame in repos_with_submodules or\
           repodir in repos_with_submodules:
            repos_with_submodules
            cmd('git submodule init')
            cmd('git submodule update')


def is_gitrepo(repo_dir):
    gitdir = join(repo_dir, '.git')
    return exists(gitdir) and isdir(gitdir)


if __name__ == '__main__':
    locals_ = locals()
    command = sys.argv[1]
    # Apply command to all repos
    gg_command(command)
