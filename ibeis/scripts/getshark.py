#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import getopt
import sys
from xml.dom.minidom import parseString
from os.path import split, splitext, join, exists, dirname
import utool as ut


def detect_sharks():
    import ibeis
    ibs = ibeis.opendb('WS_ALL')
    config = {
        'algo'            : 'yolo',
        'sensitivity'     : 0.2,
        'config_filepath' : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.cfg'),
        'weight_filepath' : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.39000.weights'),
        'class_filepath'  : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.cfg.classes'),
    }
    depc = ibs.depc_image

    imgsets = ibs.imagesets(text='Injured Sharks')
    images = ibs.images(imgsets.gids[0])
    images = images.compress([ext not in ['.gif'] for ext in images.exts])
    gid_list = images.gids

    # result is a tuple:
    # (score, bbox_list, theta_list, conf_list, class_list)
    results_list = depc.get_property('localizations', gid_list, None, config=config)

    results_list2 = []
    multi_gids = []
    failed_gids = []

    #ibs.set_image_imagesettext(failed_gids, ['Fixme'] * len(failed_gids))
    ibs.set_image_imagesettext(multi_gids, ['Fixme2'] * len(multi_gids))

    failed_gids

    for gid, res in zip(gid_list, results_list):
        score, bbox_list, theta_list, conf_list, class_list = res
        if len(bbox_list) == 0:
            failed_gids.append(gid)
        elif len(bbox_list) == 1:
            results_list2.append((gid, bbox_list, theta_list))
        elif len(bbox_list) > 1:
            multi_gids.append(gid)
            idx = conf_list.argmax()
            res2 = (gid, bbox_list[idx:idx + 1], theta_list[idx:idx + 1])
            results_list2.append(res2)

    ut.dict_hist(([t[1].shape[0] for t in results_list]))

    localized_imgs = ibs.images(ut.take_column(results_list2, 0))
    assert all([len(a) == 1 for a in localized_imgs.aids])
    old_annots = ibs.annots(ut.flatten(localized_imgs.aids))
    #old_tags = old_annots.case_tags

    # Override old bboxes
    import numpy as np
    bboxes = np.array(ut.take_column(results_list2, 1))[:, 0, :]
    ibs.set_annot_bboxes(old_annots.aids, bboxes)

    if False:
        import plottool as pt
        pt.qt4ensure()

        inter = pt.MultiImageInteraction(
            ibs.get_image_paths(ut.take_column(results_list2, 0)),
            bboxes_list=ut.take_column(results_list2, 1)
        )
        inter.dump_to_disk('shark_loc', num=50, prefix='shark_loc')
        inter.start()

        inter = pt.MultiImageInteraction(ibs.get_image_paths(failed_gids))
        inter.start()

        inter = pt.MultiImageInteraction(ibs.get_image_paths(multi_gids))
        inter.start()


def purge_ensure_one_annot_per_images(ibs):
    """
    pip install Pipe
    """
    # Purge all but one annotation
    images = ibs.images()
    #images.aids
    groups = images._annot_groups
    import numpy as np
    # Take all but the largest annotations per images
    large_masks = [ut.index_to_boolmask([np.argmax(x)], len(x)) for x in groups.bbox_area]
    small_masks = ut.lmap(ut.not_list, large_masks)
    # Remove all but the largets annotation
    small_aids = ut.zipcompress(groups.aid, small_masks)
    small_aids = ut.flatten(small_aids)

    # Fix any empty images
    images = ibs.images()
    empty_images = ut.where(np.array(images.num_annotations) == 0)
    print('empty_images = %r' % (empty_images,))
    #list(map(basename, map(dirname, images.uris_original)))

    from os.path import basename, dirname

    def VecPipe(func):
        import pipe
        @pipe.Pipe
        def wrapped(sequence):
            return map(func, sequence)
            #return (None if item is None else func(item) for item in sequence)
        return wrapped

    name_list = list(images.uris_original | VecPipe(dirname) | VecPipe(basename))
    aids_list = images.aids
    ut.assert_all_eq(list(aids_list | VecPipe(len)))
    annots = ibs.annots(ut.flatten(aids_list))
    annots.names = name_list


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
        'injury', 'net', 'hook', 'trunc', 'damage', 'scar', 'nicks', 'bite',
    ]

    injury_keys = [key for key in key_list if any([pat in key for pat in injury_patterns])]
    noninjury_keys = ut.setdiff(key_list, injury_keys)
    injury_nice = ut.lmap(lambda k: key_to_nice[k], injury_keys)  # NOQA
    noninjury_nice = ut.lmap(lambda k: key_to_nice[k], noninjury_keys)  # NOQA
    key_list = injury_keys

    keyed_images = {}
    for key in ut.ProgIter(key_list, lbl='reading index', bs=True):
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

    # Determine super-categories
    categories = ['nicks', 'scar', 'trunc']

    # Force these keys into these categories
    key_to_cat = {'scarbite': 'other_injury'}

    cat_to_keys = ut.ddict(list)

    for key in key_to_urls.keys():
        flag = 1
        if key in key_to_cat:
            cat = key_to_cat[key]
            cat_to_keys[cat].append(key)
            continue
        for cat in categories:
            if cat in key:
                cat_to_keys[cat].append(key)
                flag = 0
        if flag:
            cat = 'other_injury'
            cat_to_keys[cat].append(key)

    cat_urls = ut.ddict(list)
    for cat, keys in cat_to_keys.items():
        for key in keys:
            cat_urls[cat].extend(key_to_urls[key])

    cat_hist = {}
    for cat in list(cat_urls.keys()):
        cat_urls[cat] = list(set(cat_urls[cat]))
        cat_hist[cat] = len(cat_urls[cat])

    print(ut.repr3(cat_to_keys))
    print(ut.repr3(cat_hist))

    key_to_cat = dict([(val, key) for key, vals in cat_to_keys.items() for val in vals])

    #ingestset = {
    #    '__class__': 'ImageSet',
    #    'images': ut.ddict(dict)
    #}
    #for key, key_imgs in keyed_images.items():
    #    for imgdict in key_imgs:
    #        url = imgdict['url']
    #        encid = imgdict['correspondingEncounterNumber']
    #        # Make structure
    #        encdict = encounters[encid]
    #        encdict['__class__'] = 'Encounter'
    #        imgdict = ut.delete_keys(imgdict.copy(), ['correspondingEncounterNumber'])
    #        imgdict['__class__'] = 'Image'
    #        cat = key_to_cat[key]
    #        annotdict = {'relative_bbox': [.01, .01, .98, .98], 'tags': [cat, key]}
    #        annotdict['__class__'] = 'Annotation'

    #        # Ensure structures exist
    #        encdict['images'] = encdict.get('images', [])
    #        imgdict['annots'] = imgdict.get('annots', [])

    #        # Add an image to this encounter
    #        encdict['images'].append(imgdict)
    #        # Add an annotation to this image
    #        imgdict['annots'].append(annotdict)

    ##http://springbreak.wildbook.org/rest/org.ecocean.Encounter/1111
    #get_enc_url = 'http://www.whaleshark.org/rest/org.ecocean.Encounter/%s' % (encid,)
    #resp = requests.get(get_enc_url)
    #print(ut.repr3(encdict))
    #print(ut.repr3(encounters))

    # Download the files to the local disk
    #fpath_list =

    all_urls = ut.unique(ut.take_column(
        ut.flatten(
            ut.dict_subset(keyed_images, ut.flatten(cat_to_keys.values())).values()
        ), 'url'))

    dldir = ut.truepath('~/tmpsharks')
    from os.path import commonprefix, basename  # NOQA
    prefix = commonprefix(all_urls)
    suffix_list = [url_[len(prefix):] for url_ in all_urls]
    fname_list = [suffix.replace('/', '--') for suffix in suffix_list]

    fpath_list = []
    for url, fname in ut.ProgIter(zip(all_urls, fname_list), lbl='downloading imgs', freq=1):
        fpath = ut.grab_file_url(url, download_dir=dldir, fname=fname, verbose=False)
        fpath_list.append(fpath)

    # Make sure we keep orig info
    #url_to_keys = ut.ddict(list)
    url_to_info = ut.ddict(dict)
    for key, imgdict_list in keyed_images.items():
        for imgdict in imgdict_list:
            url = imgdict['url']
            info = url_to_info[url]
            for k, v in imgdict.items():
                info[k] = info.get(k, [])
                info[k].append(v)
            info['keys'] = info.get('keys', [])
            info['keys'].append(key)
            #url_to_keys[url].append(key)

    info_list = ut.take(url_to_info, all_urls)
    for info in info_list:
        if len(set(info['correspondingEncounterNumber'])) > 1:
            assert False, 'url with two different encounter nums'
    # Combine duplicate tags

    hashid_list = [ut.get_file_uuid(fpath_, stride=8) for fpath_ in ut.ProgIter(fpath_list, bs=True)]
    groupxs = ut.group_indices(hashid_list)[1]

    # Group properties by duplicate images
    #groupxs = [g for g in groupxs if len(g) > 1]
    fpath_list_ = ut.take_column(ut.apply_grouping(fpath_list, groupxs), 0)
    url_list_ = ut.take_column(ut.apply_grouping(all_urls, groupxs), 0)
    info_list_ = [ut.map_dict_vals(ut.flatten, ut.dict_accum(*info_))
                  for info_ in ut.apply_grouping(info_list, groupxs)]

    encid_list_ = [ut.unique(info_['correspondingEncounterNumber'])[0]
                   for info_ in info_list_]
    keys_list_ = [ut.unique(info_['keys']) for info_ in info_list_]
    cats_list_ = [ut.unique(ut.take(key_to_cat, keys)) for keys in keys_list_]

    clist = ut.ColumnLists({
        'gpath': fpath_list_,
        'url': url_list_,
        'encid': encid_list_,
        'key': keys_list_,
        'cat': cats_list_,
    })

    #for info_ in ut.apply_grouping(info_list, groupxs):
    #    info = ut.dict_accum(*info_)
    #    info = ut.map_dict_vals(ut.flatten, info)
    #    x = ut.unique(ut.flatten(ut.dict_accum(*info_)['correspondingEncounterNumber']))
    #    if len(x) > 1:
    #        info = info.copy()
    #        del info['keys']
    #        print(ut.repr3(info))

    flags = ut.lmap(ut.fpath_has_imgext, clist['gpath'])
    clist = clist.compress(flags)

    import ibeis
    ibs = ibeis.opendb('WS_Injury', allow_newdir=True)

    gid_list = ibs.add_images(clist['gpath'])
    clist['gid'] = gid_list

    failed_flags = ut.flag_None_items(clist['gid'])
    print('# failed %s' % (sum(failed_flags)),)
    passed_flags = ut.not_list(failed_flags)
    clist = clist.compress(passed_flags)
    ut.assert_all_not_None(clist['gid'])
    #ibs.get_image_uris_original(clist['gid'])
    ibs.set_image_uris_original(clist['gid'], clist['url'], overwrite=True)

    #ut.zipflat(clist['cat'], clist['key'])
    clist['tags'] = ut.zipflat(clist['cat'])
    aid_list = ibs.use_images_as_annotations(clist['gid'], adjust_percent=0.01,
                                             tags_list=clist['tags'])
    aid_list

    import plottool as pt
    from ibeis import core_annots
    pt.qt4ensure()
    #annots = ibs.annots()
    #aids = [1, 2]
    #ibs.depc_annot.get('hog', aids , 'hog')
    #ibs.depc_annot.get('chip', aids, 'img')
    for aid in ut.InteractiveIter(ibs.get_valid_aids()):
        hogs = ibs.depc_annot.d.get_hog_hog([aid])
        chips = ibs.depc_annot.d.get_chips_img([aid])
        chip = chips[0]
        hogimg = core_annots.make_hog_block_image(hogs[0])
        pt.clf()
        pt.imshow(hogimg, pnum=(1, 2, 1))
        pt.imshow(chip, pnum=(1, 2, 2))
        fig = pt.gcf()
        fig.show()
        fig.canvas.draw()

    #print(len(groupxs))

    #if False:
    #groupxs = ut.find_duplicate_items(ut.lmap(basename, suffix_list)).values()
    #print(ut.repr3(ut.apply_grouping(all_urls, groupxs)))
    #    # FIX
    #    for fpath, fname in zip(fpath_list, fname_list):
    #        if ut.checkpath(fpath):
    #            ut.move(fpath, join(dirname(fpath), fname))
    #            print('fpath = %r' % (fpath,))

    #import ibeis
    #from ibeis.dbio import ingest_dataset
    #dbdir = ibeis.sysres.lookup_dbdir('WS_ALL')
    #self = ingest_dataset.Ingestable2(dbdir)

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
