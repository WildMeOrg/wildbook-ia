# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import expanduser, normpath, realpath, join
import os
from itertools import izip
import platform


USER_ID = None
IS_USER = False
PERMITTED_REPOS = []

format_dict = {
    'https': ('.com/', 'https://'),
    'ssh': ('.com:', 'git@'),
}


def get_computer_name():
    return platform.node()


def get_repo_dirs(repo_urls, checkout_dir):
    repo_dirs = [join(checkout_dir, get_repo_dname(url)) for url in repo_urls]
    return repo_dirs


def get_repo_dname(repo_url):
    """ Break url into a dirname """
    slashpos = repo_url.rfind('/')
    colonpos = repo_url.rfind(':')
    if slashpos != -1 and slashpos > colonpos:
        pos = slashpos
    else:
        pos = colonpos
    repodir = repo_url[pos + 1 :].replace('.git', '')
    return repodir


def set_userid(userid=None, owned_computers={}, permitted_repos=[]):
    # Check to see if you are on one of Jons Computers
    global IS_USER
    global USER_ID
    global PERMITTED_REPOS
    PERMITTED_REPOS = permitted_repos
    USER_ID = userid
    IS_USER = get_computer_name() in owned_computers


def truepath(path):
    return normpath(realpath(expanduser(path)))


def unixpath(path):
    return truepath(path).replace('\\', '/')


def cd(dir_):
    dir_ = truepath(dir_)
    print('> cd ' + dir_)
    os.chdir(dir_)


def fix_repo_url(repo_url, in_type='https', out_type='ssh', format_dict=format_dict):
    """ Changes the repo_url format """
    for old, new in izip(format_dict[in_type], format_dict[out_type]):
        repo_url = repo_url.replace(old, new)
    return repo_url


def ensure_ssh_url(repo_url):
    return fix_repo_url(repo_url, in_type='https', out_type='ssh')


def repo_list(repo_urls, checkout_dir):
    repo_dirs = get_repo_dirs(repo_urls, checkout_dir)
    repo_dirs = map(unixpath, repo_dirs)
    return repo_urls, repo_dirs


def can_push(repo_url):
    owned_repo = USER_ID is not None and repo_url.find(USER_ID) != -1
    has_permit = get_repo_dname(repo_url) in PERMITTED_REPOS
    return owned_repo or has_permit


def url_list(repo_urls):
    if IS_USER:
        repo_urls = [ensure_ssh_url(url) if can_push(url) else url for url in repo_urls]
    return map(unixpath, repo_urls)


def cmd(command):
    print('> ' + command)
    os.system(command)


# def url_list2(*args):
#    """ Output is gaurenteed to be a list of paths """
#    url_list = args
#    if len(args) == 1:
#        # There is one argument
#        arg = args[0]
#        if isinstance(arg, (str, unicode)):
#            if arg.find('\n') == -1:
#                # One string long
#                url_list = [arg]
#            else:
#                # One multiline string
#                url_list = textwrap.dedent(arg).strip().split('\n')
#        else:
#            url_list = arg
#    if IS_USER:
#        def userid_in(path):
#            return IS_USER is not None and\
#                path.find(USER_ID) != -1
#        url_list = [path if userid_in(path) else fix_repo_url(path, 'https', 'ssh')
#                     for path in url_list]
#    return map(unixpath, url_list)


# def repo_list2(*args):
#    if len(args) < 1:
#        return url_list(*args)
#    elif len(args) == 2:
#        repo_urls = url_list(args[0])
#        checkout_dir = args[1]
#        repo_dirs = map(unixpath, get_repo_dirs(repo_urls, checkout_dir))
#        return repo_urls, repo_dirs
