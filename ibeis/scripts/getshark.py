#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import getopt
import sys
from xml.dom.minidom import parseString
from os.path import split, splitext, join, exists, dirname
import utool as ut


def shark_misc():
    import ibeis
    ibs = ibeis.opendb('WS_ALL')
    aid_list = ibs.get_valid_aids()
    flag_list = ibs.get_annot_been_adjusted(aid_list)
    adjusted_aids = ut.compress(aid_list, flag_list)
    return adjusted_aids


def download_sharks(XMLdata, number):
    """
    cd ~/work/WS_ALL
    python -m ibeis.scripts.getshark

    >>> from ibeis.scripts.getshark import *  # NOQA
    >>> url = 'www.whaleshark.org/listImages.jsp'
    >>> XMLdata = ut.url_read(url)
    >>> number = None
    """
    # Prepare the output directory for writing, if it doesn't exist
    output_dir = 'sharkimages'
    ut.ensuredir(output_dir)

    dom = parseString(XMLdata)

    # Download files
    if number:
        maxCount = min(number, len(dom.getElementsByTagName('img')))
    else:
        maxCount = len(dom.getElementsByTagName('img'))

    parsed_info = dict(
        img_url_list=[],
        localid_list=[],
        nameid_list=[],
        orig_fname_list=[],
        new_fname_list=[],
    )

    print('Preparing to fetch %i files...' % maxCount)

    for shark in dom.getElementsByTagName('shark'):
        localCount = 0
        for imageset in shark.getElementsByTagName('imageset'):
            for img in imageset.getElementsByTagName('img'):
                localCount += 1

                img_url = img.getAttribute('href')
                orig_fname = split(img_url)[1]
                ext = splitext(orig_fname)[1].lower()
                nameid = shark.getAttribute('number')

                new_fname = '%s-%i%s' % (
                    nameid, localCount, ext)

                parsed_info['img_url_list'].append(img_url)
                parsed_info['nameid_list'].append(nameid)
                parsed_info['localid_list'].append(localCount)
                parsed_info['orig_fname_list'].append(orig_fname)
                parsed_info['new_fname_list'].append(new_fname)

                print('Parsed %i / %i files.' % (len(parsed_info['orig_fname_list']), maxCount))

                if number is not None and len(parsed_info['orig_fname_list']) == number:
                    break
    parsed_info['new_fpath_list'] = [join(output_dir, _fname)
                                     for _fname in parsed_info['new_fname_list']]

    print('Filtering parsed images')

    # Filter based on image type (keep only jpgs)
    ext_flags = [_fname.endswith('.jpg') or _fname.endswith('.jpg')
                  for _fname in parsed_info['new_fname_list']]
    parsed_info = {key: ut.compress(list_, ext_flags) for key, list_ in parsed_info.items()}

    # Filter to only images matching the appropriate tags
    from ibeis import tag_funcs
    parsed_info['tags_list'] = parse_shark_tags(parsed_info['orig_fname_list'])
    tag_flags = tag_funcs.filterflags_general_tags(
        parsed_info['tags_list'],
        has_any=['view-left'],
        none_match=['qual.*', 'view-top', 'part-.*', 'cropped'],
    )
    parsed_info = {key: ut.compress(list_, tag_flags) for key, list_ in parsed_info.items()}
    print('Tags in chosen images:')
    print(ut.dict_hist(ut.flatten(parsed_info['tags_list'] )))

    # Download selected subset
    print('Downloading selected subset')
    _iter = list(zip(parsed_info['img_url_list'],
                     parsed_info['new_fpath_list']))
    _iter = ut.ProgressIter(_iter, lbl='downloading sharks')
    for img_url, new_fpath in _iter:
        if not exists(new_fpath):
            ut.download_url(img_url, new_fpath)

    # Remove corrupted or ill-formatted images
    print('Checking for corrupted images')
    import vtool as vt
    noncorrupt_flags = vt.filterflags_valid_images(parsed_info['new_fpath_list'])
    parsed_info = {
        key: ut.compress(list_, noncorrupt_flags)
        for key, list_ in parsed_info.items()
    }

    print('Removing small images')
    import numpy as np
    imgsize_list = np.array([vt.open_image_size(gpath) for gpath in parsed_info['new_fpath_list']])
    sqrt_area_list = np.sqrt(np.prod(imgsize_list, axis=1))
    areq_flags_list = sqrt_area_list >= 750
    parsed_info = {key: ut.compress(list_, areq_flags_list)
                   for key, list_ in parsed_info.items()}

    grouped_idxs = ut.group_items(list(range(len(parsed_info['nameid_list']))),
                                  parsed_info['nameid_list'])
    keep_idxs = sorted(ut.flatten([idxs for key, idxs in grouped_idxs.items() if len(idxs) >= 2]))
    parsed_info = {key: ut.take(list_, keep_idxs) for key, list_ in parsed_info.items()}

    print('Moving imagse to secondary directory')
    named_outputdir = 'named-left-sharkimages'
    # Build names
    parsed_info['namedir_fpath_list'] = [
        join(named_outputdir, _nameid, _fname)
        for _fname, _nameid in zip(parsed_info['new_fname_list'],
                                   parsed_info['nameid_list'])]
    # Create directories
    ut.ensuredir(named_outputdir)
    named_dirs = ut.unique_ordered(list(map(dirname, parsed_info['namedir_fpath_list'])))
    for dir_ in named_dirs:
        ut.ensuredir(dir_)
    # Copy
    ut.copy_files_to(src_fpath_list=parsed_info['new_fpath_list'],
                     dst_fpath_list=parsed_info['namedir_fpath_list'])


def parse_shark_tags(orig_fname_list):
    import re

    invalid_tag_patterns = [
        re.escape('-'),
        re.escape('(') + '?\\d*' + re.escape(')') + '?',
        '\\d+-\\d+-\\d+', '\\d+,',
        '\\d+', 'vi*', 'i*v', 'i+',
        '\\d+th', '\\d+nd', '\\d+rd',
        'remant', 'timnfe', 't', 'e', 'sjl', 'disc', 'dec', 'road', 'easter',
        'western', 'west', 'tn',
        '\\d*ap',
        'whaleshark\\d*', 'shark\\d*', 'whale\\d*',
        'whalesharking', 'sharking', 'whalesharks', 'whales',
        'picture',
        'australien',
        'australia',
        'nick', 'tim\\d*',
        'imageset',
        'holiday', 'visit', 'tour', 'trip', 'pec', 'sv',
        'a', 'b',
        'gender', 'sex',
        'img', 'image', 'pic', 'pics', 'leith', 'trips', 'kings', 'photo', 'video', 'media',
        'fix', 'feeding',
        'nrd', 'nd', 'gen', 'wa', 'nmp', 'bo', 'kd', 'ow', 'ne', 'dsc', 'nwd',
        'mg', 'w', 'mai', 'blue', 'stumpy',
        'oea', 'cbe', 'edc', 'knrt',
        'tiws2',
        'ando', 'adv', 'str', 'adventure',
        'camera', 'tag', 'id',
        'of', 'and',
        'tagged', 'from',
        'day', '\\d*april', '\\d*may', '\\d*july', '\\d*june',
        'ningaloo', 'ningblue\\d*', 'kooling',
    ]

    valid_tag_level_set = [
        ['view-left', 'left', 'lhs', 'l', 'leftside'],
        ['view-right', 'right', 'rhs', 'r', 'rightside'],
        ['view-back', 'back'],
        ['view-top', 'top'],
        ['sex-male', 'male', 'm', 'sexm'],
        ['sex-female', 'female', 'f'],
        ['sex-unknown', 'unknown', 'u'],
        ['part-tail', 'tail'],
        ['part-flank', 'side', 'flank'],
        ['part-head', 'head'],
        ['part-pectoral', 'pectoral', 'pec'],
        ['part-dorsal', 'dorsal', 'dorsals'],
        ['part-claspers', 'claspers', 'clasper'],
        ['part-fin', 'fin'],
        ['cropped', 'crop'],
        ['scar', 'scar2'],
        ['notch'],
        ['small'],
        ['bite'],
        ['cam-slr2', 'slr2'],
        #['cam-5m', '5m']
        ['5m'],
        ['7m'],
        ['4m'],
        ['copy'],
        ['qual-resize'],
        ['qual-stretched'],
    ]

    def apply_enum_regex(pat_list):
        enum_endings = [
            '[a-g]',
            '\\d*',
            'i*',
        ]
        expanded_pats = ut.flatten([
            [pat + end for end in enum_endings]
            for pat  in pat_list
        ])
        return expanded_pats

    def apply_regex_endings(pat_list):
        return [p + '$' for p in pat_list]

    tag_alias_map = {}
    for level_set in valid_tag_level_set:
        main_key = level_set[0]
        for key in level_set:
            tag_alias_map[key] = main_key

    inverse_alias_map = {}
    for level_set in valid_tag_level_set:
        inverse_alias_map[level_set[0]] = level_set

    regex_alias_map = {
        'view-left': apply_regex_endings(apply_enum_regex(inverse_alias_map['view-left'])),
        'view-right': apply_regex_endings(apply_enum_regex(inverse_alias_map['view-right'])),
    }

    valid_tags = list(inverse_alias_map.keys())

    invalid_tag_patterns = apply_regex_endings(invalid_tag_patterns)

    def parse_all_fname_tags(fname):
        _tags = [splitext(fname)[0]]
        _tags = ut.flatten([t.split('_') for t in _tags])
        _tags = ut.flatten([t.split('.') for t in _tags])
        _tags = [t.lower() for t in _tags]
        _tags = [tag_alias_map.get(t, t) for t in _tags]
        for key, vals in regex_alias_map.items():
            pat = ut.regex_or(vals)
            _tags = [key if re.match(pat, t) else t for t in _tags]
        pat = ut.regex_or(invalid_tag_patterns)
        _tags = [t for t in _tags if not re.match(pat, t)]
        _tags = ut.unique_ordered(_tags)
        return _tags

    all_img_tag_list = list(map(parse_all_fname_tags, orig_fname_list))

    known_img_tag_list = [list(set(tags).intersection(set(valid_tags)))
                          for tags in all_img_tag_list]

    if False:
        # Help figure out which tags are important
        _parsed_tags = ut.flatten(all_img_tag_list)

        taghist =  ut.dict_hist(_parsed_tags)
        taghist = {key: val for key, val in taghist.items() if val > 1}

        unknown_taghist = sorted([
            (val, key) for key, val in taghist.items()
            if key not in valid_tags
        ])[::-1]
        known_taghist = sorted([
            (val, key) for key, val in taghist.items()
            if key in valid_tags
        ])[::-1]

        print('Known')
        print(ut.list_str(known_taghist[0:100]))

        print('Unknown')
        print(ut.list_str(unknown_taghist[0:100]))

        print(ut.dict_str(
            ut.dict_hist(ut.flatten(known_img_tag_list)),
            key_order_metric='val'
        ))

    return known_img_tag_list


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:u:n:h')
    except getopt.GetoptError:
        usage()
        sys.exit(1)

    filename = None
    url = 'www.whaleshark.org/listImages.jsp'
    number = 0

    # Handle command-line arguments
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-f':
            filename = arg
        elif opt == '-u':
            url = arg
        elif opt == '-n':
            try:
                number = int(arg)
            except ValueError:
                usage()
                sys.exit()

    # Open the XML file and extract its contents as a DOM object
    if filename:
        XMLdata = ut.readfrom(filename)
    else:
        XMLdata = ut.url_read(url)
        #with open('XMLData.xml', 'w') as file_:
        #    file_.write(XMLdata)
    print('Downloading')
    download_sharks(XMLdata, number)


def usage():
    print('Fetches a number of images from the ECOCEAN shark database.')
    print('Options:')
    print('  -f <FILENAME> - Reads XML data from a file, rather than a URL.')
    print('  -u <URL> - Reads XML data from the given URL.')
    print('  -n <NUMBER> - Number of images to read; if omitted, reads all of them.')
    print('  -h - Prints this help text.')


if __name__ == '__main__':
    main()
