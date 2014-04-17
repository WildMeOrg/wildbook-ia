from __future__ import absolute_import, division, print_function
from os.path import exists, join, realpath, dirname
import utool
from utool import util_arg, util_cache

args = None

WORKDIR_CACHEID = 'work_directory_cache_id'


# TODO: workdir doesnt belong in params
def get_workdir(allow_gui=True):
    work_dir = util_cache.global_cache_read(WORKDIR_CACHEID, default='.')
    if work_dir is not '.' and exists(work_dir):
        return work_dir
    if allow_gui:
        work_dir = set_workdir()
        return get_workdir(allow_gui=False)
    return None


def set_workdir(work_dir=None, allow_gui=True):
    if work_dir is None and allow_gui:
        work_dir = guiselect_workdir()
    if work_dir is None or not exists(work_dir):
        raise AssertionError('invalid workdir=%r' % work_dir)
    util_cache.global_cache_write(WORKDIR_CACHEID, work_dir)


def guiselect_workdir():
    import guitool
    guitool.ensure_qtapp()
    # Gui selection
    work_dir = guitool.select_directory('Work dir not currently set.' +
                                        'Select a work directory')
    # Make sure selection is ok
    if not exists(work_dir):
        try_again = guitool.user_option(
            paremt=None,
            msg='Directory %r does not exist.' % work_dir,
            title='get work dir failed',
            options=['Try Again'],
            use_cache=False)
        if try_again == 'Try Again':
            return guiselect_workdir()
    return work_dir


def get_dbalias_dict():
    dbalias_dict = {}
    computer_name = utool.util_cplat.get_computer_name()
    if computer_name in ['Hyrule', 'Ooo', 'BakerStreet']:
        # For jon's convinience
        dbalias_dict.update({
            'NAUTS':            'NAUT_Dan',
            'WD':               'WD_Siva',
            'LF':               'LF_all',
            'GZ':               'GZ_ALL',
            'MOTHERS':          'PZ_MOTHERS',
            'FROGS':            'Frogs',
            'TOADS':            'WY_Toads',
            'SEALS':            'Seals',

            'OXFORD':           'Oxford_Buildings',
            'PARIS':            'Paris_Buildings',

            'JAG_KELLY':        'JAG_Kelly',
            'JAG_KIERYN':       'JAG_Kieryn',
            'WILDEBEAST':       'Wildebeast',
            'WDOGS':            'WD_Siva',

            'PZ':               'PZ_FlankHack',
            'PZ2':              'PZ-Sweatwater',
            'PZ_MARIANNE':      'PZ_Marianne',
            'PZ_DANEXT_TEST':   'PZ_DanExt_Test',
            'PZ_DANEXT_ALL':    'PZ_DanExt_All',

            'LF_ALL':           'LF_all',
            'WS_HARD':          'WS_hard',
            'SONOGRAMS':        'sonograms',

        })
        dbalias_dict['JAG'] = dbalias_dict['JAG_KELLY']
    return dbalias_dict


def db_to_dbdir(db, allow_newdir=False):
    import os
    import sys
    import utool
    work_dir = get_workdir()
    dbdir = join(work_dir, db)
    dbalias_dict = get_dbalias_dict()
    if not exists(dbdir) and db.upper() in dbalias_dict:
        dbdir = join(work_dir, dbalias_dict[db.upper()])

    if not exists(dbdir) and allow_newdir:
        if exists(dirname(dbdir)):
            print('[params] MAKE DIR: %r' % dbdir)
            os.mkdir(dbdir)

    if not exists(dbdir):
        print('!!!!!!!!!!!!!!!!!!!!!')
        print('[params] WARNING: db=%r not found in work_dir=%r' %
              (db, work_dir))
        fname_list = os.listdir(work_dir)
        lower_list = [fname.lower() for fname in fname_list]
        index = utool.listfind(lower_list, db.lower())
        if index is not None:
            print('[params] WARNING: db capitalization seems to be off')
            if not '--strict' in sys.argv:
                print('[params] attempting to fix it')
                db = fname_list[index]
                dbdir = join(work_dir, db)
                print('[params] dbdir=%r' % dbdir)
                print('[params] db=%r' % db)
        if not exists(dbdir):
            print('[params] Valid DBs:')
            print('\n'.join(fname_list))
            print('[params] dbdir=%r' % dbdir)
            print('[params] db=%r' % db)
            print('[params] work_dir=%r' % work_dir)

            raise AssertionError('[params] FATAL ERROR. Cannot load database')
        print('!!!!!!!!!!!!!!!!!!!!!')
    return dbdir


def parse_args(defaultdb='cache', allow_newdir=False, **kwargs):
    # TODO: Port more from hotspotter/hsdev/argparse2.py
    # TODO: Incorporate kwargs
    global args
    parser2 = util_arg.make_argparse2('IBEIS - lite', version='???')

    def dev_argparse(parser2):
        parser2.add_str(('--tests', '--test', '-t'),  [], 'integer or test name', nargs='*')

    def behavior_argparse(parser2):
        # Program behavior
        parser2 = parser2.add_argument_group('Behavior')
        # TODO UNFILTER THIS HERE AND CHANGE PARALLELIZE TO KNOW ABOUT
        # MEMORY USAGE
        parser2.add_int('--num-procs', default=None, help='defaults util_parallel.init_pools method')
        parser2.add_flag('--serial', help='Forces num_procs=1')
        parser2.add_flag('--nogui', help='Will not start the gui')
        parser2.add_int('--loop-freq', default=100, help='Qt main loop ms frequency')
        parser2.add_flag('--cmd', help='Runs in IPython mode')
        parser2.add_flag('--nocache-db', help='Disables db cache')
        parser2.add_flag('--nocache-flann', help='Disables flann cache')
        parser2.add_flag('--nocache-query', help='Disables flann cache')
        parser2.add_flag('--auto-dump', help='dumps the SQLITE3 database after every commit')

    def database_argparse(parser2):
        # Database selections
        parser2 = parser2.add_argument_group('Database')
        parser2.add_str('--db', defaultdb, 'specifies the short name of the database to load')
        parser2.add_str('--dbdir', None, 'specifies the full path of the database to load')

    behavior_argparse(parser2)
    database_argparse(parser2)
    dev_argparse(parser2)

    args, unknown = parser2.parser.parse_known_args()

    # Apply any argument dependencies here
    def postprocess(args):
        if args.serial:
            args.num_proces = 1
        if args.dbdir is not None:
            # The full path is specified
            args.dbdir = realpath(args.dbdir)
        if args.dbdir is None or args.dbdir in ['', ' ', '.'] or not exists(args.dbdir):
            args.dbdir = None
        if args.dbdir is None:
            if args.db is not None:
                if args.db == 'cache':
                    if not args.nocache_db:
                        # Read dbdir from cache
                        args.dbdir = util_cache.global_cache_read('cached_dbdir', default=None)
                else:
                    args.dbdir = db_to_dbdir(args.db, allow_newdir=allow_newdir)
        return args
    args = postprocess(args)
