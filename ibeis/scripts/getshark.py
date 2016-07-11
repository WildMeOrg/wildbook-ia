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


def get_injured_sharks():
    """
    >>> from ibeis.scripts.getshark import *  # NOQA
    """
    import requests
    url = 'http://www.whaleshark.org/getKeywordImages.jsp'
    resp = requests.get(url)
    assert resp.status_code == 200
    keywords = resp.json()['keywords']
    key_list = ut.take_column(keywords, 'indexName')
    key_to_nice  = {k['indexName']: k['readableName'] for k in keywords}

    injury_patterns = [
        'injury',
        'net',
        'hook',
        'trunc',
        'damage',
        'scar',
        'nicks',
        'bite',
    ]

    injury_keys = [key for key in key_list if any([pat in key for pat in injury_patterns])]
    noninjury_keys = ut.setdiff(key_list, injury_keys)
    injury_nice = ut.lmap(lambda k: key_to_nice[k], injury_keys)  # NOQA
    noninjury_nice = ut.lmap(lambda k: key_to_nice[k], noninjury_keys)  # NOQA

    key_list = injury_keys

    keyed_images = {}

    for key in ut.ProgIter(key_list, lbl='reading index'):
        key_url = url + '?indexName={indexName}'.format(indexName=key)
        key_resp = requests.get(key_url)
        assert key_resp.status_code == 200
        key_imgs = key_resp.json()['images']
        keyed_images[key] = key_imgs

    key_hist = {key: len(imgs) for key, imgs in keyed_images.items()}
    key_hist = ut.sort_dict(key_hist, ut.identity)
    print(ut.repr3(key_hist))
    nice_key_hist = ut.map_dict_keys(lambda k: key_to_nice[k], key_hist)
    nice_key_hist = ut.sort_dict(nice_key_hist, ut.identity)
    print(ut.repr3(nice_key_hist))

    key_to_urls = {key: ut.take_column(vals, 'url') for key, vals in keyed_images.items()}
    overlaps = {}
    import itertools
    overlap_img_list = []
    for k1, k2 in itertools.combinations(key_to_urls.keys(), 2):
        overlap_imgs = ut.isect(key_to_urls[k1], key_to_urls[k2])
        num_overlap = len(overlap_imgs)
        overlaps[(k1, k2)] = num_overlap
        overlaps[(k1, k1)] = len(key_to_urls[k1])
        if num_overlap > 0:
            #print('[%s][%s], overlap=%r' % (k1, k2, num_overlap))
            overlap_img_list.extend(overlap_imgs)

    all_img_urls = list(set(ut.flatten(key_to_urls.values())))
    num_all = len(all_img_urls)  # NOQA
    print('num_all = %r' % (num_all,))

    key_to_category = ut.ddict(list)
    cat_urls = ut.ddict(list)
    categories = ['nicks', 'scar', 'trunc']
    for key in key_to_urls.keys():
        flag = 1
        for cat in categories:
            if cat in key:
                key_to_category[cat].append(key)
                cat_urls[cat] += key_to_urls[key]
                flag = 0
        if flag:
            cat = 'other'
            key_to_category[cat].append(key)
            cat_urls[cat] += key_to_urls[key]

    cat_hist = {}
    for cat in list(cat_urls.keys()):
        cat_urls[cat] = list(set(cat_urls[cat]))
        cat_hist[cat] = len(cat_urls[cat])

    print(ut.repr3(key_to_category))
    print(ut.repr3(cat_hist))

    if False:
        # Show overlap matrix
        import plottool as pt
        import pandas as pd
        import numpy as np
        dict_ = overlaps
        s = pd.Series(dict_, index=pd.MultiIndex.from_tuples(overlaps))
        df = s.unstack()
        lhs, rhs = df.align(df.T)
        df = lhs.add(rhs, fill_value=0).fillna(0)

        label_texts = df.columns.values

        def label_ticks(label_texts):
            import plottool as pt
            truncated_labels = [repr(lbl[0:100]) for lbl in label_texts]
            ax = pt.gca()
            ax.set_xticks(list(range(len(label_texts))))
            ax.set_xticklabels(truncated_labels)
            [lbl.set_rotation(-55) for lbl in ax.get_xticklabels()]
            [lbl.set_horizontalalignment('left') for lbl in ax.get_xticklabels()]

            #xgrid, ygrid = np.meshgrid(range(len(label_texts)), range(len(label_texts)))
            #pt.plot_surface3d(xgrid, ygrid, disjoint_mat)
            ax.set_yticks(list(range(len(label_texts))))
            ax.set_yticklabels(truncated_labels)
            [lbl.set_horizontalalignment('right') for lbl in ax.get_yticklabels()]
            [lbl.set_verticalalignment('center') for lbl in ax.get_yticklabels()]
            #[lbl.set_rotation(20) for lbl in ax.get_yticklabels()]

        #df = df.sort(axis=0)
        #df = df.sort(axis=1)

        sortx = np.argsort(df.sum(axis=1).values)[::-1]
        df = df.take(sortx, axis=0)
        df = df.take(sortx, axis=1)

        fig = pt.figure(fnum=1)
        fig.clf()
        mat = df.values.astype(np.int32)
        mat[np.diag_indices(len(mat))] = 0
        vmax = mat[(1 - np.eye(len(mat))).astype(np.bool)].max()
        import matplotlib.colors
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax, clip=True)
        pt.plt.imshow(mat, cmap='hot', norm=norm, interpolation='none')
        pt.plt.colorbar()
        pt.plt.grid('off')
        label_ticks(label_texts)
        fig.tight_layout()

    #overlap_df = pd.DataFrame.from_dict(overlap_img_list)

    class TmpImage(ut.NiceRepr):
        pass

    key = bigkey = list(key_hist.keys())[-1]
    url_list = ut.take_column(keyed_images[bigkey], 'url')
    for url in ut.ProgIter(url_list, lbl='downloading imgs', freq=1):
        fpath = ut.grab_file_url(url, download_dir='.')

    from skimage.feature import hog
    from skimage import data, color, exposure
    import plottool as pt
    image2 = color.rgb2gray(data.astronaut())  # NOQA

    fpath = './GOPR1120.JPG'

    import vtool as vt
    for fpath in [fpath]:
        """
        http://scikit-image.org/docs/dev/auto_examples/plot_hog.html
        """

        image = vt.imread(fpath, grayscale=True)
        image = pt.color_funcs.to_base01(image)

        fig = pt.figure(fnum=2)
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)

        fig, (ax1, ax2) = pt.plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=pt.plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=pt.plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        pt.plt.show()

    #for


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
