# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import ubelt as ub
import utool as ut
import warnings

(print, rrr, profile) = ut.inject2(__name__)


def have_gpu(min_memory=8000):
    """ Determine if we are on a machine with a good GPU """
    # FIXME: HACK
    gpus = gpu_info()
    if not gpus:
        return False
    return any(gpu['mem_total'] >= min_memory for gpu in gpus)
    # import platform
    # GPU_MACHINES = {'arisia', 'aretha'}
    # # Maybe we look at nvidia-smi instead?
    # hostname = platform.node()
    # return hostname in GPU_MACHINES


def find_unused_gpu(min_memory=0):
    """
    Finds GPU with the lowest memory usage by parsing output of nvidia-smi

    python -c "from pysseg.util import gpu_util; print(gpu_util.find_unused_gpu())"
    """
    gpus = gpu_info()
    if gpus is None:
        return None
    gpu_avail_mem = {n: gpu['mem_avail'] for n, gpu in gpus.items()}
    usage_order = ub.argsort(gpu_avail_mem)
    gpu_num = usage_order[-1]
    if gpu_avail_mem[gpu_num] < min_memory:
        return None
    else:
        return gpu_num


def gpu_info():
    """
    Parses nvidia-smi
    """
    result = ub.cmd('nvidia-smi')
    if result['ret'] != 0:
        warnings.warn('Could not run nvidia-smi.')
        return None

    lines = result['out'].splitlines()

    gpu_lines = []
    current = None

    for line in lines:
        if current is None:
            # Signals the start of GPU info
            if line.startswith('|====='):
                current = []
        else:
            if len(line.strip()) == 0:
                # End of GPU info
                break
            elif line.startswith('+----'):
                # Move to the next GPU
                gpu_lines.append(current)
                current = []
            else:
                current.append(line)

    def parse_gpu_lines(lines):
        line1 = lines[0]
        line2 = lines[1]
        gpu = {}
        gpu['name'] = ' '.join(line1.split('|')[1].split()[1:-1])
        gpu['num'] = int(' '.join(line1.split('|')[1].split()[0]))

        mempart = line2.split('|')[2].strip()
        part1, part2 = mempart.split('/')
        gpu['mem_used'] = float(part1.strip().replace('MiB', ''))
        gpu['mem_total'] = float(part2.strip().replace('MiB', ''))
        gpu['mem_avail'] = gpu['mem_total'] - gpu['mem_used']
        return gpu

    gpus = {}
    for num, lines in enumerate(gpu_lines):
        gpu = parse_gpu_lines(lines)
        assert (
            num == gpu['num']
        ), 'nums ({}, {}) do not agree. probably a parsing error'.format(num, gpu['num'])
        assert (
            num not in gpus
        ), 'Multiple GPUs labeled as num {}. Probably a parsing error'.format(num)
        gpus[num] = gpu
    return gpus
