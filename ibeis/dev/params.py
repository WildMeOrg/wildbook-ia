args = None


def parse_args(**kwargs):
    # TODO: Port more from hotspotter/hsdev/argparse2.py
    # TODO: Incorporate kwargs
    global args
    from ibeis.dev import argparse2
    import multiprocessing
    parser2 = argparse2.make_argparse2('IBEIS - lite', version='???')

    def behavior_argparse(parser2):
        # Program behavior
        parser2 = parser2.add_argument_group('Behavior')
        # TODO UNFILTER THIS HERE AND CHANGE PARALLELIZE TO KNOW ABOUT
        # MEMORY USAGE
        num_cpus = max(min(6, multiprocessing.cpu_count()), 1)
        num_proc_help = 'default to number of cpus = %d' % (num_cpus)
        parser2.add_int('--num-procs', num_cpus, num_proc_help)
        parser2.add_flag('--serial', help='Forces num_procs=1')
        parser2.add_flag('--nogui', help='Will not start the gui')
        parser2.add_int('--loop-freq', default=100, help='Qt main loop ms frequency')
        parser2.add_flag('--cmd', help='Runs in IPython mode')

    def database_argparse(parser2):
        # Database selections
        parser2 = parser2.add_argument_group('Database')
        parser2.add_str('--db', 'DEFAULT', 'specifies the short name of the database to load')
        parser2.add_str('--dbdir', None, 'specifies the full path of the database to load')

    behavior_argparse(parser2)
    database_argparse(parser2)
    args, unknown = parser2.parser.parse_known_args()
