#!/usr/bin/env python
"""
Removes profiled output of code that never ran
"""
from __future__ import absolute_import, division, print_function
from six.moves import range
import six
import sys
import operator
import utool

"""
input_fname = 'raw_profile.dev.py.2014-09-23_18-28-57.raw.prof'
"""


def __dbg_list(list_):

    for item in list_:
        newline = item.find('\n')
        twoline = item[newline + 1:].find('\n')
        head = item[0:(newline + twoline)]
        print(head)


def get_block_totaltime(block):
    time_line = utool.regex_search('Total time: [0-9.]* s', block)
    time_str  = utool.regex_search('[0-9.]+', time_line)
    if time_str is not None:
        return float(time_str)
    else:
        return None


def clean_line_profile_text(text):
    """
    Sorts the output from line profile by execution time
    Removes entries which were not run
    """
    #
    # Split the file into blocks along delimters and and put delimeters back in the list
    delim = 'Total time: '
    #delim = 'File: '
    list_ = utool.regex_split('^' + delim, text)
    for ix in range(1, len(list_)):
        list_[ix] = delim + list_[ix]
    #
    # Build a map from times to line_profile blocks
    prefix_list = []
    timemap = utool.ddict(list)
    for ix in range(len(list_)):
        block = list_[ix]
        total_time = get_block_totaltime(block)
        # Blocks without time go at the front of sorted output
        if total_time is None:
            prefix_list.append(block)
        # Blocks that are not run are not appended to output
        elif total_time != 0:
            timemap[total_time].append(block)
    # Sort the blocks by time
    sorted_lists = sorted(six.iteritems(timemap), key=operator.itemgetter(0))
    newlist = prefix_list[:]
    for key, val in sorted_lists:
        newlist.extend(val)
    # Rejoin output text
    output_text = '\n'.join(newlist)
    return output_text


def clean_lprof_file(input_fname, output_fname=None):
    """ Reads a .lprof file and cleans it """
    # Read the raw .lprof text dump
    with open(input_fname) as file_:
        text = file_.read()
    # Sort and clean the text
    output_text = clean_line_profile_text(text)
    return output_text


if __name__ == '__main__':
    # Only profiled functions that are run are printed
    print('[profile_cleaner] __main__')
    input_fname = sys.argv[1]
    output_fname = sys.argv[2] if len(sys.argv) > 2 else None
    print('[profile_cleaner] cleaning')
    output_text = clean_lprof_file(input_fname, output_fname)
    print('[profile_cleaner] dumping')
    if output_fname is not None:
        # Output to file
        with open(output_fname, 'w') as file2_:
            file2_.write(output_text)
    else:
        print(output_text)
