from __future__ import absolute_import, division, print_function
# Python
import psutil
import os
from .util_inject import inject
from .util_str import byte_str2
print, print_, printDBG, rrr, profile = inject(__name__, '[print]')


def time_str2(seconds):
    return '%.2f sec' % (seconds,)


def time_in_usermode():
    import resource
    stime = resource.getrusage(resource.RUSAGE_SELF).ru_stime
    return stime


def time_in_systemmode():
    import resource
    utime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    return utime


def peak_memory():
    """Returns the resident set size (the portion of
    a process's memory that is held in RAM.)
    """
    import resource
    # MAXRSS is expressed in kilobytes. Convert to bytes
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return maxrss


def print_resource_usage():
    print('+______________________')
    print('|    RESOURCE_USAGE    ')
    print('|  * current_memory = %s' % byte_str2(current_memory_usage()))
    #print('|  * peak_memory    = %s' % byte_str2(peak_memory()))
    #print('|  * user_time      = %s' % time_str2(time_in_usermode()))
    #print('|  * system_time    = %s' % time_str2(time_in_systemmode()))
    print('L______________________')


def get_resource_limits():
    import resource
    #rlimit_keys = [key for key in resource.__dict__.iterkeys() if key.startswith('RLIMIT_')]
    #print('\n'.join(['(\'%s\', resource.%s),' % (key.replace('RLIMIT_', ''), key) for key in rlimit_keys]))
    rlim_keytups = [
        ('MEMLOCK', resource.RLIMIT_MEMLOCK),
        ('NOFILE', resource.RLIMIT_NOFILE),
        ('CPU', resource.RLIMIT_CPU),
        ('DATA', resource.RLIMIT_DATA),
        ('OFILE', resource.RLIMIT_OFILE),
        ('STACK', resource.RLIMIT_STACK),
        ('FSIZE', resource.RLIMIT_FSIZE),
        ('CORE', resource.RLIMIT_CORE),
        ('NPROC', resource.RLIMIT_NPROC),
        ('AS', resource.RLIMIT_AS),
        ('RSS', resource.RLIMIT_RSS),
    ]
    rlim_valtups = [(lbl, resource.getrlimit(rlim_key)) for (lbl, rlim_key) in rlim_keytups]
    def rlimval_str(rlim_val):
        soft, hard = rlim_val
        softstr = byte_str2(soft) if soft != -1 else 'None'
        hardstr = byte_str2(hard) if hard != -1 else 'None'
        return '%12s, %12s' % (softstr, hardstr)
    rlim_strs = ['%8s: %s' % (lbl, rlimval_str(rlim_val)) for (lbl, rlim_val) in rlim_valtups]
    print('Resource Limits: ')
    print('%8s  %12s  %12s' % ('id', 'soft', 'hard'))
    print('\n'.join(rlim_strs))
    return rlim_strs


#def rusage_flags():
    #0	ru_utime	time in user mode (float)
    #1	ru_stime	time in system mode (float)
    #2	ru_maxrss	maximum resident set size
    #3	ru_ixrss	shared memory size
    #4	ru_idrss	unshared memory size
    #5	ru_isrss	unshared stack size
    #6	ru_minflt	page faults not requiring I/O
    #7	ru_majflt	page faults requiring I/O
    #8	ru_nswap	number of swap outs
    #9	ru_inblock	block input operations
    #10	ru_oublock	block output operations
    #11	ru_msgsnd	messages sent
    #12	ru_msgrcv	messages received
    #13	ru_nsignals	signals received
    #14	ru_nvcsw	voluntary context switches
    #15	ru_nivcsw	involuntary context switches


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


#psutil.virtual_memory()
#psutil.swap_memory()
#psutil.disk_partitions()
#psutil.disk_usage("/")
#psutil.disk_io_counters()
#psutil.net_io_counters(pernic=True)
#psutil.get_users()
#psutil.get_boot_time()
#psutil.get_pid_list()
