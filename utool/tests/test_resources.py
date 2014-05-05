#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
#from utool import util_resources
import psutil
import os


if __name__ == '__main__':
    meminfo = psutil.Process(os.getpid()).get_memory_info()
    rss = meminfo[0]  # Resident Set Size / Mem Usage
    vms = meminfo[1]  # Virtual Memory Size / VM Size  # NOQA
    print('+-----------------------')
    print('|  BEFORE UTOOL IMPORT  ')
    print('|  * current_memory (before import) = %.2f MB' % (rss / (2.0 ** 20)))
    print('importing')
    import utool
    utool.print_resource_usage()
