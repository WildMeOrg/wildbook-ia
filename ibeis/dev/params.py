args = None


def parse_args(**kwargs):
    # TODO: Port more from hotspotter/hsdev/argparse2.py
    # TODO: Incorporate kwargs
    global args
    from utool import util_arg
    parser2 = util_arg.make_argparse2('IBEIS - lite', version='???')

    def behavior_argparse(parser2):
        # Program behavior
        parser2 = parser2.add_argument_group('Behavior')
        # TODO UNFILTER THIS HERE AND CHANGE PARALLELIZE TO KNOW ABOUT
        # MEMORY USAGE
        parser2.add_int('--num-procs', default=None, help='defaults to max number of cpus')
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

    # Apply any argument dependencies here
    def postprocess(args):
        if args.serial:
            args.num_proces = 1
        return args
    args = postprocess(args)
