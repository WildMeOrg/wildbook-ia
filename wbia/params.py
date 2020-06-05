# -*- coding: utf-8 -*-
"""
DEPRICATE THIS ENTIRE FILE

this module lists most of the command line args available for use.  there are
still many cases where util_arg.get_argval and util_arg.get_argflag are used
instead of this module. Those command line arguments will not be represented
here and they should eventually be integrated into this module (hopefully
automagically)

TODO:
nnkj/enerate this module automagically from
    import utool as ut
    utool_parse_codeblock = ut.util_arg.autogen_argparse_block(extra_args=parsed_args)
    ut.util_arg.reset_argrecord()
    import wbia
    parsed_args = ut.util_arg.parse_used_arg_flags_and_vals(wbia, recursive=True)
    wbia_parse_codeblock = ut.util_arg.autogen_argparse_block(extra_args=parsed_args)

    ut.util_arg.autogenerate_parse_py([utool_parse_codeblock, wbia_parse_codeblock])

    utool_parse_codeblock
    ut.util_arg

    print(parse_codeblock)
"""
from __future__ import absolute_import, division, print_function
from utool import util_arg
import utool as ut
import os

(print, rrr, profile) = ut.inject2(__name__)


# Global command line arguments
args = None  # Parsed arguments
unknown = None  # Unparsed arguments


def parse_args():
    # TODO: Port more from hotspotter/hsdev/argparse2.py
    global args
    global unknown
    if args is not None:
        # Only parse arguments once
        if util_arg.VERBOSE:
            print('[!params] ALREADY INITIALIZED ARGS')
        return
    program_name = 'IBEIS - Lite (WARNING THESE ARGS ARE MOSTLY DEPRICATED)'
    description = 'Image Based Ecological Information System'
    parser2 = util_arg.make_argparse2(program_name, description)

    def dev_argparse(parser2):
        parser2 = parser2.add_argument_group('Developer')
        parser2.add_str(
            ('--tests', '--test', '-t'), [], help='integer or test name', nargs='*'
        )
        parser2.add_flag(('--wait', '-w'), help='wait for user to press enter')
        parser2.add_flag(('--cmd', '--ipy'), help='Runs in IPython mode')
        parser2.add_flag(('--all-cases', '--all'))
        parser2.add_flag(
            ('--all-gt-cases', '--allgt'),
            help='chooses all groundtruthed annotations to be queried',
        )
        parser2.add_flag(('--all-singleton-cases', '--allsingle'))
        parser2.add_ints(
            ('--qindex', '-qx', '--index'),
            None,
            help='test only these query indices. Out of bounds errors are clipped',
        )
        parser2.add_ints(
            ('--dindex', '-dx'),
            None,
            help='test only these database indices. . Out of bounds errors are clipped',
        )
        # parser2.add_ints(('--sel-rows', '-r'), help='view row for experiment harness')
        # parser2.add_ints(('--sel-cols', '-c'), help='view col for experiment harness')
        # parser2.add_ints(('--qaid', '--qaids'), default=[], help='investigate match aid')
        # parser2.add_ints(('--daid-exclude', '--daids-exclude'), default=[], help='exclude daids from matching')
        # parser2.add_flag(('--convert'), help='converts / updates schema of database if possible')
        # parser2.add_flag(('--force-delete'), help='forces deletion of hsdb before convert')
        parser2.add_flag(
            ('--fulltb'),
            help='shows a full traceback (default behavior removes decorators from the trace)',
        )
        parser2.add_flag(('--verbose'), help='turns on verbosity')
        parser2.add_flag(
            (('--veryverbose', '--very-verbose')), help='turns on extra verbosity'
        )
        parser2.add_flag(('--quiet'), help='turns down verbosity')
        parser2.add_flag(('--silent'), help='turns off verbosity')
        parser2.add_flag(
            ('--print-inject-order'),
            help='shows import order of any module registered with utool',
        )
        parser2.add_flag(
            ('--debug-print'), help='shows where each injected print statement happens'
        )

    def behavior_argparse(parser2):
        # Program behavior
        parser2 = parser2.add_argument_group('Behavior')
        parser2.add_int(
            '--num-procs', default=None, help='defaults util_parallel.init_pools method'
        )
        parser2.add_flag('--serial', help='Forces num_procs=1')
        parser2.add_flag('--nogui', default=False, help='Will not start the gui.')
        parser2.add_flag('--gui', default=True, help='Will start the gui if able.')
        loopfreq = 4200 / 10  # 100
        parser2.add_int('--loop-freq', default=loopfreq, help='Qt main loop ms frequency')
        # parser2.add_flag('--nocache-db',
        #                 help='Disables db cache')
        # parser2.add_flag('--nocache-flann',
        #                 help='Disables flann cache')
        # parser2.add_flag('--nocache-query',
        #                 help='Disables flann cache')
        parser2.add_flag(
            '--auto-dump', help='dumps the SQLITE3 database after every commit'
        )
        # parser2.add_flag('--darken')
        parser2.add_flag('--aggroflush', help='utool writes flush immediately')
        parser2.add_flag('--nologging', help='disables logging')
        parser2.add_flag('--noindent', help='disables utool indentation')
        # parser2.add_str('--wildbook-target', help='specify the Wildbook target deployment')

    def database_argparse(parser2):
        # Database selections
        parser2 = parser2.add_argument_group('Database')
        parser2.add_str(
            '--db', None, help='specifies the short name of the database to load'
        )
        parser2.add_str(
            '--dbdir', None, help='specifies the full path of the database to load'
        )
        parser2.add_str('--set-workdir', None)
        parser2.add_flag('--get-workdir', help='gets the default work directory')
        parser2.add_str(
            ('--logdir', '--set-logdir'), None, help='sets the default logging directory',
        )
        parser2.add_flag('--get-logdir', help='gets the current logging directory')
        parser2.add_flag(
            ('--view-logdir', '--vld'),
            help='views the current (local and global) logging directories',
        )
        parser2.add_flag(
            ('--view-logdir-local', '--vldl'),
            help='views the current local logging directory',
        )
        parser2.add_flag(
            ('--view-logdir-global', '--vldg'),
            help='views the current global logging directory',
        )
        parser2.add_flag(
            '--force-incremental-db-update',
            help='ignores the current database schema and forces an incremental update for new databases',
        )
        parser2.add_flag(
            '--dump-autogen-schema',
            help='dumps (autogenerates) the current database schema based on the expected versions in the controller',
        )

    def commands_argparse(parser2):
        parser2 = parser2.add_argument_group('Commands')
        parser2.add_flag(
            ('--set-default-dbdir', '--setdb'),
            help='sets the opening database to be the default',
        )
        parser2.add_flag('--dump-global-cache')
        parser2.add_flag('--dump-argv')
        # parser2.add_flag('--gvim-notes')
        parser2.add_flag(
            ('--view-database-directory', '--vdd'), help='opens the database directory'
        )

        # NEED TO DEPCIRATE THIS VERY BADLY
        parser2.add_strs(
            ('--update-query-cfg', '--set-cfg', '--cfg'),
            default=None,
            help=(
                'set query parameters from the commandline: e.g. '
                '--cfg xy_thresh=.01 score_method=csum'
            ),
        )
        parser2.add_flag(
            ('--preload-exit', '--prequit', '--prele'),
            help='exit after preload commands',
        )
        parser2.add_flag(
            ('--postload-exit', '--postquit', '--postle'),
            help='exit after postload commands',
        )
        parser2.add_flag(
            ('--webapp', '--webapi', '--web', '--browser'),
            help='automatically launch the web app / web api',
        )
        parser2.add_int(
            ('--webport', '--web-port', '--port'),
            help='specify the port for the web api',
            default=None,
        )

    def postload_gui_commands_argparse(parser2):
        parser2 = parser2.add_argument_group('Postload GUI Commands')
        parser2.add_int(('--select-nid', '--nid'), help='view col')
        parser2.add_int(('--select-gid', '--gid'), help='view col')
        parser2.add_int(('--select-aid', '--aid'), help='view col')
        parser2.add_ints(('--query-aid', '--query'), help='query aid(s)')
        parser2.add_flag(('--edit-notes'), help='edits database notes')
        parser2.add_str(
            ('--set-all-species'), help='careful. overwrites all species info.'
        )
        parser2.add_flag(
            ('--dump-schema', '--print-schema'), help='dumps schema to stdout'
        )
        parser2.add_flag(('--delete-cache'), help='deletes most of the cache')
        parser2.add_flag(('--delete-cache-complete'), help='deletes all cached data')
        parser2.add_flag(
            ('--delete-query-cache', '--delete-qres-cache', '--clear_qres'),
            help='deletes the query result cache',
        )

    behavior_argparse(parser2)
    database_argparse(parser2)
    dev_argparse(parser2)
    commands_argparse(parser2)
    postload_gui_commands_argparse(parser2)

    args, unknown = parser2.parser.parse_known_args()

    # Apply any argument postprocessing dependencies here
    args.gui = not args.nogui
    if args.serial:
        args.num_procs = 1


# Dont parse args if environment variable is off
# We use this to turn off arg parsing when Sphinx is running
if (
    os.environ.get('IBIES_PARSE_ARGS', 'ON') == 'ON'
    and os.environ.get('UTOOL_AUTOGEN_SPHINX_RUNNING', 'OFF') == 'OFF'
):
    parse_args()
