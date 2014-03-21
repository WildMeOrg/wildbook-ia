from __future__ import division, print_function
import sys
import time
import datetime
from .util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[time]')


# --- Timing ---
def tic(msg=None):
    return (msg, time.time())


def toc(tt):
    (msg, start_time) = tt
    ellapsed = (time.time() - start_time)
    if not msg is None:
        sys.stdout.write('...toc(%.4fs, ' % ellapsed + '"' + str(msg) + '"' + ')\n')
    return ellapsed


def get_timestamp(format_='filename', use_second=False):
    now = datetime.datetime.now()
    if use_second:
        time_tup = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        time_formats = {
            'filename': 'ymd_hms-%04d-%02d-%02d_%02d-%02d-%02d',
            'comment': '# (yyyy-mm-dd hh:mm:ss) %04d-%02d-%02d %02d:%02d:%02d'}
    else:
        time_tup = (now.year, now.month, now.day, now.hour, now.minute)
        time_formats = {
            'filename': 'ymd_hm-%04d-%02d-%02d_%02d-%02d',
            'comment': '# (yyyy-mm-dd hh:mm) %04d-%02d-%02d %02d:%02d'}
    stamp = time_formats[format_] % time_tup
    return stamp


class Timer(object):
    ''' Timer with-statment context object
    e.g with Timer() as t: some_function()'''
    def __init__(self, msg='', verbose=True, newline=True):
        self.msg = msg
        self.verbose = verbose
        self.newline = newline
        self.tstart = -1
        self.tic()

    def tic(self):
        if self.verbose:
            sys.stdout.flush()
            print_('\ntic(%r)' % self.msg)
            if self.newline:
                print_('\n')
            sys.stdout.flush()
        self.tstart = time.time()

    def toc(self):
        ellapsed = (time.time() - self.tstart)
        if self.verbose:
            print_('...toc(%r)=%.4fs\n' % (self.msg, ellapsed))
            sys.stdout.flush()
        return ellapsed

    def __enter__(self):
        #if not self.msg is None:
            #sys.stdout.write('---tic---'+self.msg+'  \n')
        #self.tic()
        pass

    def __exit__(self, type, value, trace):
        self.toc()
