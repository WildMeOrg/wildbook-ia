from __future__ import absolute_import, division, print_function
import sys
from .util_inject import inject
from .util_arg import get_flag
print, print_, printDBG, rrr, profile = inject(__name__, '[progress]')


QUIET = get_flag('--quiet')
VERBOSE = get_flag('--verbose')
VALID_PROGRESS_TYPES = ['none', 'dots', 'fmtstr', 'simple']


def simple_progres_func(verbosity, msg, progchar='.'):
    def mark_progress0(*args):
        pass

    def mark_progress1(*args):
        sys.stdout.write(progchar)

    def mark_progress2(*args):
        print(msg % args)

    if verbosity == 0:
        mark_progress = mark_progress0
    elif verbosity == 1:
        mark_progress = mark_progress1
    elif verbosity == 2:
        mark_progress = mark_progress2
    return mark_progress


def prog_func(*args, **kwargs):
    return progress_func(*args, **kwargs)


# TODO: Return start_prog, make_prog, end_prog
def progress_func(max_val=0, lbl='Progress: ', mark_after=-1,
                  flush_after=4, spacing=0, line_len=80,
                  progress_type='fmtstr', mark_start=False, repl=False):
    '''Returns a function that marks progress taking the iteration count as a
    parameter. Prints if max_val > mark_at. Prints dots if max_val not
    specified or simple=True'''
    write_fn = sys.stdout.write
    #write_fn = print_
    #print('STARTING PROGRESS: VERBOSE=%r QUIET=%r' % (VERBOSE, QUIET))

    # Tell the user we are about to make progress
    if QUIET or (progress_type in ['simple', 'fmtstr'] and max_val < mark_after):
        return lambda count: None, lambda: None
    # none: nothing
    if progress_type == 'none':
        mark_progress =  lambda count: None
    # simple: one dot per progress. no flush.
    if progress_type == 'simple':
        mark_progress = lambda count: write_fn('.')
    # dots: spaced dots
    if progress_type == 'dots':
        indent_ = '    '
        write_fn(indent_)

        if spacing > 0:
            # With spacing
            newline_len = spacing * line_len // spacing

            def mark_progress_sdot(count):
                write_fn('.')
                count_ = count + 1
                if (count_) % newline_len == 0:
                    write_fn('\n' + indent_)
                    sys.stdout.flush()
                elif (count_) % spacing == 0:
                    write_fn(' ')
                    sys.stdout.flush()
                elif (count_) % flush_after == 0:
                    sys.stdout.flush()
            mark_progress = mark_progress_sdot
        else:
            # No spacing
            newline_len = line_len

            def mark_progress_dot(count):
                write_fn('.')
                count_ = count + 1
                if (count_) % newline_len == 0:
                    write_fn('\n' + indent_)
                    sys.stdout.flush()
                elif (count_) % flush_after == 0:
                    sys.stdout.flush()
            mark_progress = mark_progress_dot
    # fmtstr: formated string progress
    if progress_type == 'fmtstr':
        fmt_str = progress_str(max_val, lbl=lbl, repl=repl)

        def mark_progress_fmtstr(count):
            count_ = count + 1
            write_fn(fmt_str % (count_))
            if (count_) % flush_after == 0:
                sys.stdout.flush()
        mark_progress = mark_progress_fmtstr
    # FIXME idk why argparse2.ARGS_ is none here.
    if '--aggroflush' in sys.argv:
        def mark_progress_agressive(count):
            mark_progress(count)
            sys.stdout.flush()
        return mark_progress_agressive

    def end_progress():
        write_fn('\n')
        sys.stdout.flush()
    #mark_progress(0)
    if mark_start:
        mark_progress(-1)
    return mark_progress, end_progress
    raise Exception('unkown progress type = %r' % progress_type)


def progress_str(max_val, lbl='Progress: ', repl=False):
    r'makes format string that prints progress: %Xd/MAX_VAL with backspaces'
    max_str = str(max_val)
    dnumstr = str(len(max_str))
    cur_str = '%' + dnumstr + 'd'
    if repl:
        fmt_str = lbl.replace('<cur_str>', cur_str).replace('<max_str>', max_str)
    else:
        fmt_str = lbl + cur_str + '/' + max_str
    fmt_str = '\b' * (len(fmt_str) - len(dnumstr) + len(max_str)) + fmt_str
    return fmt_str
