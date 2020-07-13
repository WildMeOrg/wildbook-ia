# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import sys

(print, rrr, profile) = ut.inject2(__name__)


def hack_argv(arg):
    # HACK COMMON ARGV SYMBOLS
    if arg.startswith('--hargv='):
        hack_argv_key = '='.join(arg.split('=')[1:])

        common_args = [
            '--dpath=~/latex/crall-thesis-2017/',
            '--clipwhite',
            '--contextadjust',
            '--dpi=256',
        ]

        sys.argv.extend(common_args)

        if '--noshow' not in sys.argv:
            sys.argv.extend(['--diskshow'])

        # Figsize hacks

        if hack_argv_key in ['surf']:
            sys.argv.extend(
                [
                    '--figsize=14,3',
                    '--hspace=.3',
                    # '--top=.8',
                    '--top=.85',
                    '--bottom=0.18',
                    '--left=.05',
                    '--right=.95',
                ]
            )
        elif hack_argv_key in ['scores']:
            sys.argv.extend(
                [
                    # '--figsize=15,7',
                    # '--top=.8',
                    '--figsize=14,3',
                    '--top=.8',
                    '--hspace=.3',
                    '--bottom=0.08',
                    '--left=.05',
                    '--right=.95',
                ]
            )
        elif hack_argv_key in ['tags']:
            sys.argv.extend(
                [
                    # '--figsize=14,3',
                    # '--top=.8',
                    # '--hspace=.3',
                    # '--bottom=0.08',
                    # '--left=.05',
                    # '--right=.95'
                ]
            )
        elif hack_argv_key in ['expt']:
            sys.argv.extend(
                [
                    # '--figsize=15,3',
                    '--figsize=15,3.1',
                    '--top=.9',
                    '--bottom=.15',
                ]
            )
        elif hack_argv_key in ['mech']:
            sys.argv.extend(
                [
                    # '--figsize=14,5',
                    '--figsize=14,3',
                    '--top=.9',
                ]
            )
        elif hack_argv_key in ['tags']:
            sys.argv.extend(['--bottom=.3', '--left=.2'])

        # Save location
        # fname_fmt = 'figuresX/expt_{e}_{db}_a_{a}_t_{t}'
        fname_fmt = 'figuresX/expt_{label}'

        # if hack_argv_key in ['scores']:
        #    fname_fmt += '_{filt}'

        # if hack_argv_key in ['time']:
        #    if not ('--falsepos' in sys.argv) or ('--truepos' in sys.argv):
        #        fname_fmt += '_TP'
        #    if ('--falsepos' in sys.argv):
        #        fname_fmt += '_FP'

        if hack_argv_key in ['time', 'expt', 'mech', 'scores', 'surf', 'tags']:
            sys.argv.extend(['--save', fname_fmt + '.png'])

        if hack_argv_key in ['time']:
            sys.argv.extend(['--figsize=18,8', r'--width=".8\textwidth"'])


# import IPython
# IPython.embed()
for arg in sys.argv[:]:
    hack_argv(arg)

import utool as ut  # NOQA

# ut.show_if_requested()
# sys.exit(1)
from utool.util_six import get_funcname  # NOQA

# import functools

# A list of registered development test functions
DEVCMD_FUNCTIONS = []
DEVPRECMD_FUNCTIONS = []

# DEVCMD_FUNCTIONS2 = []


# def devcmd2(*args):
#    """ Decorator which registers a function as a developer command """
#    noargs = len(args) == 1 and not isinstance(args[0], str)
#    if noargs:
#        # Function is only argument
#        func = args[0]
#        func_aliases = []
#    else:
#        func_aliases = list(args)
#    def closure_devcmd2(func):
#        global DEVCMD_FUNCTIONS2
#        func_aliases.extend([get_funcname(func)])
#        DEVCMD_FUNCTIONS2.append((tuple(func_aliases), func))
#        def func_wrapper(*args_, **kwargs_):
#            #if ut.VERBOSE:
#            #if ut.QUIET:
#            print('[DEVCMD2] ' + ut.func_str(func, args_, kwargs_))
#            return func(*args_, **kwargs_)
#        return func_wrapper
#    if noargs:
#        return closure_devcmd2(func)
#    return closure_devcmd2


def devcmd(*args):
    """ Decorator which registers a function as a developer command """
    noargs = len(args) == 1 and not isinstance(args[0], str)
    if noargs:
        # Function is only argument
        func = args[0]
        func_aliases = []
    else:
        func_aliases = list(args)

    def closure_devcmd(func):
        global DEVCMD_FUNCTIONS
        func_aliases.extend([get_funcname(func)])
        DEVCMD_FUNCTIONS.append((tuple(func_aliases), func))

        def func_wrapper(*args_, **kwargs_):
            # if ut.VERBOSE:
            # if ut.QUIET:
            print('[DEVCMD] ' + ut.func_str(func, args_, kwargs_))
            return func(*args_, **kwargs_)

        return func_wrapper

    if noargs:
        return closure_devcmd(func)
    return closure_devcmd


def devprecmd(*args):
    """ Decorator which registers a function as a developer precommand """
    noargs = len(args) == 1 and not isinstance(args[0], str)
    if noargs:
        # Function is only argument
        func = args[0]
        func_aliases = []
    else:
        func_aliases = list(args)

    def closure_devprecmd(func):
        global DEVPRECMD_FUNCTIONS
        func_aliases.extend([get_funcname(func)])
        DEVPRECMD_FUNCTIONS.append((tuple(func_aliases), func))

        def func_wrapper(*args_, **kwargs_):
            # if ut.VERBOSE:
            # if ut.QUIET:
            print('[DEVPRECMD] ' + ut.func_str(func, args_, kwargs_))
            return func(*args_, **kwargs_)

        return func_wrapper

    if noargs:
        return closure_devprecmd(func)
    return closure_devprecmd
