from __future__ import absolute_import, division, print_function
from utool import util_arg

# Global command line arguments
args = None     # Parsed arguments
unknown = None  # Unparsed arguments


def parse_args():
    # TODO: Port more from hotspotter/hsdev/argparse2.py
    global args
    global unknown
    if args is not None:
        # Only parse arguments once
        print('[!params] ALREADY INITIALIZED ARGS')
        return None
    program_name = 'IBEIS - Lite'
    description = 'Image Based Ecological Information System'
    parser2 = util_arg.make_argparse2(program_name, description)

    def dev_argparse(parser2):
        parser2 = parser2.add_argument_group('Developer')
        parser2.add_str(('--tests', '--test', '-t'),  [],
                        help='integer or test name', nargs='*')
        parser2.add_flag(('--wait', '-w'),  help='wait for user to press enter')
        parser2.add_flag(('--cmd', '--ipy'), help='Runs in IPython mode')
        parser2.add_flag(('--all-cases', '--all'))
        parser2.add_flag(('--all-gt-cases', '--allgt'))
        parser2.add_flag(('--all-hard-cases', '--allhard'))  # all_hard_cases
        parser2.add_flag(('--all-singleton-cases', '--allsingle'))
        parser2.add_ints(('--index', '-x'), None, help='test only this index')
        parser2.add_ints(('--sel-rows', '-r'), help='view row')
        parser2.add_ints(('--sel-cols', '-c'), help='view col')
        parser2.add_ints('--qaid', default=[], help='investigate match aid')
        parser2.add_ints('--daid-exclude', default=[], help='exclude daids from matching')
        parser2.add_flag(('--convert'), help='converts / updates schema of database if possible')
        parser2.add_flag(('--force-delete'), help='forces deletion of hsdb before convert')
        parser2.add_flag(('--fulltb'), help='shows a full traceback (default behavior removes decorators from the trace)')
        parser2.add_str(('--merge-species'), help='merges all databases of given species')

    def behavior_argparse(parser2):
        # Program behavior
        parser2 = parser2.add_argument_group('Behavior')
        parser2.add_int('--num-procs', default=None,
                        help='defaults util_parallel.init_pools method')
        parser2.add_flag('--serial', help='Forces num_procs=1')
        parser2.add_flag('--nogui', default=False,
                         help='Will not start the gui.')
        parser2.add_flag('--gui', default=True,
                         help='Will start the gui if able.')
        loopfreq = 4200 / 10  # 100
        parser2.add_int('--loop-freq', default=loopfreq,
                        help='Qt main loop ms frequency')
        parser2.add_flag('--nocache-db',
                         help='Disables db cache')
        parser2.add_flag('--nocache-flann',
                         help='Disables flann cache')
        parser2.add_flag('--nocache-query',
                         help='Disables flann cache')
        parser2.add_flag('--auto-dump',
                         help='dumps the SQLITE3 database after every commit')
        parser2.add_flag('--darken')
        parser2.add_flag('--aggroflush', help='utool writes flush immediately')
        parser2.add_flag('--nologging', help='diables logging')

    def database_argparse(parser2):
        # Database selections
        parser2 = parser2.add_argument_group('Database')
        parser2.add_str('--db', None,
                        help='specifies the short name of the database to load')
        parser2.add_str('--dbdir', None,
                        help='specifies the full path of the database to load')
        parser2.add_str('--set-workdir', None)
        parser2.add_flag('--get-workdir', help='gets the default work directory')
        parser2.add_str(('--logdir', '--set-logdir'), None,
                        help='sets the default logging directory')
        parser2.add_flag('--force-incremental-db-update',
                         help='ignores the current database schema and forces an incremental update for new databases')
        parser2.add_flag('--dump-autogen-schema',
                         help='dumps (autogenerates) the current database schema based on the expected versions in the controller')

    def commands_argparse(parser2):
        parser2 = parser2.add_argument_group('Commands')
        parser2.add_flag(('--set-default-dbdir', '--setdb'),
                         help='sets the opening database to be the default')
        parser2.add_flag('--dump-global-cache')
        parser2.add_flag('--dump-argv')
        #parser2.add_flag('--gvim-notes')
        parser2.add_flag(('--view-database-directory', '--vdd'),
                         help='opens the database directory')
        parser2.add_strs(('--update-query-cfg', '--set-cfg', '--cfg'), default=None,
                         help=('set query parameters from the commandline: e.g. '
                               '--cfg xy_thresh=.01 score_method=csum'))
        parser2.add_flag(('--preload-exit', '--prequit', '--prele'), help='exit after preload commands')
        parser2.add_flag(('--postload-exit', '--postquit', '--postle'), help='exit after postload commands')
        parser2.add_flag(('--webapp', '--webapi', '--web', '--browser'), help='automatically launch the web app / web api')

    def postload_gui_commands_argparse(parser2):
        parser2 = parser2.add_argument_group('Postload GUI Commands')
        parser2.add_int(('--select-nid', '--nid'), help='view col')
        parser2.add_int(('--select-gid', '--gid'), help='view col')
        parser2.add_int(('--select-aid', '--aid'), help='view col')
        parser2.add_int(('--query-aid', '--query'), help='query aid')
        parser2.add_flag(('--edit-notes'), help='edits database notes')
        parser2.add_str(('--set-notes'), help='overwrites database notes')
        parser2.add_ints('--set-aids-as-hard', help='set hard tag on selected aids')
        parser2.add_str(('--set-all-species'), help='careful. overwrites all species info.')
        parser2.add_flag(('--dump-schema', '--print-schema'), help='dumps schema to stdout')
        parser2.add_flag(('--delete-cache'), help='deletes the cache')

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


parse_args()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
