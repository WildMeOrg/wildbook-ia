from __future__ import absolute_import, division, print_function
# Python
import psutil
import os
from .util_inject import inject
from .util_str import byte_str2
print, print_, printDBG, rrr, profile = inject(__name__, '[print]')


def peak_memory():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def current_memory_usage():
    meminfo = psutil.Process(os.getpid()).get_memory_info()
    rss = meminfo[0]  # Resident Set Size / Mem Usage
    vms = meminfo[1]  # Virtual Memory Size / VM Size  # NOQA
    return rss


def num_cpus():
    return psutil.NUM_CPUS


def available_memory():
    return psutil.virtual_memory().available


def total_memory():
    return psutil.virtual_memory().total


def used_memory():
    return total_memory() - available_memory()


def memstats():
    print('[psutil] total     = %s' % byte_str2(total_memory()))
    print('[psutil] available = %s' % byte_str2(available_memory()))
    print('[psutil] used      = %s' % byte_str2(used_memory()))
    print('[psutil] current   = %s' % byte_str2(current_memory_usage()))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    memstats()


#psutil.virtual_memory()
#psutil.swap_memory()
#psutil.disk_partitions()
#psutil.disk_usage("/")
#psutil.disk_io_counters()
#psutil.net_io_counters(pernic=True)
#psutil.get_users()
#psutil.get_boot_time()
#psutil.get_pid_list()
