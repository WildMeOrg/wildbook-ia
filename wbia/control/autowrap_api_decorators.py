#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import os
import logging

import utool as ut

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')

BLACKLIST = ['get_images', 'get_annotations', 'get_chips']


def get_func(line):
    index = line.find('(')
    line = line[4:index]
    return line


def get_parts(line, sub):
    line = get_func(line)
    if line.startswith('_'):
        logger.info('Processing Line: {!r}'.format(line))
        logger.info('    Stripped: {!r}'.format(line))
        input('SKIPPED')
        return None, None, None
    if line in BLACKLIST:
        logger.info('Processing Line: {!r}'.format(line))
        logger.info('    Stripped: {!r}'.format(line))
        input('BLACKLISTED')
        return None, None, None
    # Ascertain method
    if line == 'delete_{}'.format(sub):
        return sub, '', 'delete'
    if line == 'delete_{}s'.format(sub):
        return sub, '', 'delete'
    if '_valid_' in line and 'rowid' not in line:
        return sub, '', 'get'
    if line.startswith('get_'):
        line = line.replace('get_', '')
        method = 'get'
    elif line.startswith('set_'):
        line = line.replace('set_', '')
        method = 'put'
    elif line.startswith('update_'):
        line = line.replace('update_', '')
        method = 'put'
    elif line.startswith('delete_'):
        line = line.replace('delete_', '')
        method = 'delete'
    elif line.startswith('add_'):
        line = line.replace('add_', '')
        method = 'post'
        return sub, '', method
    else:
        logger.info('Processing Line: {!r}'.format(line))
        logger.info('    Stripped: {!r}'.format(line))
        input('FAILED')
        return None, None, None
    # logger.info('    Method-less: %r' % (line, ))
    submodule = sub
    # logger.info('    Submodule: %r' % (submodule, ))
    line = line.replace('{}s_'.format(submodule), '')
    line = line.replace('{}_'.format(submodule), '')
    line = line.replace('_{}s'.format(submodule), '')
    line = line.replace('_{}'.format(submodule), '')
    func = line
    # logger.info('    Function: %r' % (func, ))
    return submodule, func, method


def get_decorator(submodule, func, method):
    if submodule is None or func is None or method is None:
        return None
    if len(func) > 0:
        url = '/api/{}/{}/'.format(submodule, func)
    else:
        url = '/api/{}/'.format(submodule)
    method = method.upper()
    return url, method


def process_file(filename, sub):
    filename_src = '{}.py'.format(filename)
    filename_dst = '{}_processed.py'.format(filename)
    # filename_cmp = '%s_manual.py' % (filename, )
    # Open source file
    with open(filename_src, 'r') as src:
        lines = src.read()
    lines = lines.replace(':\n    """ ', ':\n    r"""\n    ')
    lines = lines.replace(":\n    ''' ", ":\n    r'''\n    ")
    lines = lines.replace(':\n    """', ':\n    r"""')
    lines = lines.replace(":\n    '''", ":\n    r'''")
    lines = ['{}\n'.format(line) for line in lines.split('\n')]

    # Expand docs
    preprocessed = []
    func = None
    latest = None
    multiline = False
    for line in lines:
        if multiline:
            preprocessed.append(line)
            if ':' in line:
                multiline = False
            continue
        if func is not None:
            if line.count('"""') == 2 or line.count("'''") == 2:
                line_ = line.strip()
                line_ = line_.replace('"""', '')
                line_ = line_.replace("'''", '')
                line_ = line_.strip()
                preprocessed.append('    r"""\n')
                preprocessed.append('    {}\n'.format(line_))
                preprocessed.append('    """\n')
                func = None
                continue
            elif line.count('"""') == 0 and line.count("'''") == 0:
                preprocessed.append('    r"""\n')
                preprocessed.append('    Auto-docstr for {!r}\n'.format(func))
                preprocessed.append('    """\n')
                preprocessed.append(line)
                func = None
                continue
        if line.endswith('"""\n') and line.strip() != '"""' and line.strip() != 'r"""':
            line = line.replace(' """\n', '\n')
            preprocessed.append(line)
            preprocessed.append('    """\n')
            continue
        # if line.endswith("'''\n") and line.strip() != "'''" and line.strip() != "r'''":
        #     line = line.replace(" '''\n", "\n")
        #     preprocessed.append(line)
        #     preprocessed.append("    '''\n")
        #     continue
        if line.startswith('def'):
            func = get_func(line)
            preprocessed.append(line)
            if ':' not in line:
                multiline = True
            continue
        preprocessed.append(line)
        func = None

    # Process lines
    processed = []
    incomment = False
    latest = None
    for line in preprocessed[:-1]:
        if incomment:
            if latest is not None and line == '    """\n':
                url, method = latest
                processed.append('\n')
                processed.append('    RESTful:\n')
                processed.append('        Method: {}\n'.format(method))
                processed.append('        URL:    {}\n'.format(url))
                incomment = False
            if latest is not None and (
                line == '    Example:\n' or line == '    Example0:\n'
            ):
                url, method = latest
                processed.append('    RESTful:\n')
                processed.append('        Method: {}\n'.format(method))
                processed.append('        URL:    {}\n'.format(url))
                processed.append('\n')
                incomment = False
        if line == '    r"""\n':
            incomment = True
        if line.startswith('def'):
            submodule, func, method = get_parts(line, sub)
            latest = get_decorator(submodule, func, method)
            if latest is not None:
                url, method = latest
                wrapper = "@register_api('{}', methods=['{}'])\n".format(url, method)
                # logger.info(wrapper)
                processed.append(wrapper)
        processed.append(line)

    # Write destination file
    with open(filename_dst, 'w') as dst:
        dst.write(''.join(processed))

    # output = os.popen('diff %s %s' % (filename_dst, filename_cmp, )).read()
    # logger.info(output)


if __name__ == '__main__':
    # filename = 'manual_image_funcs'
    # submodule = 'image'
    filename = 'manual_meta_funcs'
    # submodule = filename.split('_')[1]
    submodule = 'contributor'
    process_file(filename, submodule)
