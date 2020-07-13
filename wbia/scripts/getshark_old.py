# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from os.path import join, dirname, basename

(print, rrr, profile) = ut.inject2(__name__)


def get_injured_sharks():
    """
    >>> from wbia.scripts.getshark import *  # NOQA
    """
    import requests

    url = 'http://www.whaleshark.org/getKeywordImages.jsp'
    resp = requests.get(url)
    assert resp.status_code == 200
    keywords = resp.json()['keywords']
    key_list = ut.take_column(keywords, 'indexName')
    key_to_nice = {k['indexName']: k['readableName'] for k in keywords}

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

    injury_keys = [
        key for key in key_list if any([pat in key for pat in injury_patterns])
    ]
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
    key_hist = ut.sort_dict(key_hist, 'vals')
    print(ut.repr3(key_hist))
    nice_key_hist = ut.map_dict_keys(lambda k: key_to_nice[k], key_hist)
    nice_key_hist = ut.sort_dict(nice_key_hist, 'vals')
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
            # print('[%s][%s], overlap=%r' % (k1, k2, num_overlap))
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

    # ingestset = {
    #    '__class__': 'ImageSet',
    #    'images': ut.ddict(dict)
    # }
    # for key, key_imgs in keyed_images.items():
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

    # # http://springbreak.wildbook.org/rest/org.ecocean.Encounter/1111
    # get_enc_url = 'http://www.whaleshark.org/rest/org.ecocean.Encounter/%s' % (encid,)
    # resp = requests.get(get_enc_url)
    # print(ut.repr3(encdict))
    # print(ut.repr3(encounters))

    # Download the files to the local disk
    # fpath_list =

    all_urls = ut.unique(
        ut.take_column(
            ut.flatten(
                ut.dict_subset(keyed_images, ut.flatten(cat_to_keys.values())).values()
            ),
            'url',
        )
    )

    dldir = ut.truepath('~/tmpsharks')
    from os.path import commonprefix, basename  # NOQA

    prefix = commonprefix(all_urls)
    suffix_list = [url_[len(prefix) :] for url_ in all_urls]
    fname_list = [suffix.replace('/', '--') for suffix in suffix_list]

    fpath_list = []
    for url, fname in ut.ProgIter(
        zip(all_urls, fname_list), lbl='downloading imgs', freq=1
    ):
        fpath = ut.grab_file_url(url, download_dir=dldir, fname=fname, verbose=False)
        fpath_list.append(fpath)

    # Make sure we keep orig info
    # url_to_keys = ut.ddict(list)
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
            # url_to_keys[url].append(key)

    info_list = ut.take(url_to_info, all_urls)
    for info in info_list:
        if len(set(info['correspondingEncounterNumber'])) > 1:
            assert False, 'url with two different encounter nums'
    # Combine duplicate tags

    hashid_list = [
        ut.get_file_uuid(fpath_, stride=8) for fpath_ in ut.ProgIter(fpath_list, bs=True)
    ]
    groupxs = ut.group_indices(hashid_list)[1]

    # Group properties by duplicate images
    # groupxs = [g for g in groupxs if len(g) > 1]
    fpath_list_ = ut.take_column(ut.apply_grouping(fpath_list, groupxs), 0)
    url_list_ = ut.take_column(ut.apply_grouping(all_urls, groupxs), 0)
    info_list_ = [
        ut.map_dict_vals(ut.flatten, ut.dict_accum(*info_))
        for info_ in ut.apply_grouping(info_list, groupxs)
    ]

    encid_list_ = [
        ut.unique(info_['correspondingEncounterNumber'])[0] for info_ in info_list_
    ]
    keys_list_ = [ut.unique(info_['keys']) for info_ in info_list_]
    cats_list_ = [ut.unique(ut.take(key_to_cat, keys)) for keys in keys_list_]

    clist = ut.ColumnLists(
        {
            'gpath': fpath_list_,
            'url': url_list_,
            'encid': encid_list_,
            'key': keys_list_,
            'cat': cats_list_,
        }
    )

    # for info_ in ut.apply_grouping(info_list, groupxs):
    #    info = ut.dict_accum(*info_)
    #    info = ut.map_dict_vals(ut.flatten, info)
    #    x = ut.unique(ut.flatten(ut.dict_accum(*info_)['correspondingEncounterNumber']))
    #    if len(x) > 1:
    #        info = info.copy()
    #        del info['keys']
    #        print(ut.repr3(info))

    flags = ut.lmap(ut.fpath_has_imgext, clist['gpath'])
    clist = clist.compress(flags)

    import wbia

    ibs = wbia.opendb('WS_Injury', allow_newdir=True)

    gid_list = ibs.add_images(clist['gpath'])
    clist['gid'] = gid_list

    failed_flags = ut.flag_None_items(clist['gid'])
    print('# failed %s' % (sum(failed_flags)),)
    passed_flags = ut.not_list(failed_flags)
    clist = clist.compress(passed_flags)
    ut.assert_all_not_None(clist['gid'])
    # ibs.get_image_uris_original(clist['gid'])
    ibs.set_image_uris_original(clist['gid'], clist['url'], overwrite=True)

    # ut.zipflat(clist['cat'], clist['key'])
    if False:
        # Can run detection instead
        clist['tags'] = ut.zipflat(clist['cat'])
        aid_list = ibs.use_images_as_annotations(
            clist['gid'], adjust_percent=0.01, tags_list=clist['tags']
        )
        aid_list

    import wbia.plottool as pt
    from wbia import core_annots

    pt.qt4ensure()
    # annots = ibs.annots()
    # aids = [1, 2]
    # ibs.depc_annot.get('hog', aids , 'hog')
    # ibs.depc_annot.get('chip', aids, 'img')
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

    # print(len(groupxs))

    # if False:
    # groupxs = ut.find_duplicate_items(ut.lmap(basename, suffix_list)).values()
    # print(ut.repr3(ut.apply_grouping(all_urls, groupxs)))
    #    # FIX
    #    for fpath, fname in zip(fpath_list, fname_list):
    #        if ut.checkpath(fpath):
    #            ut.move(fpath, join(dirname(fpath), fname))
    #            print('fpath = %r' % (fpath,))

    # import wbia
    # from wbia.dbio import ingest_dataset
    # dbdir = wbia.sysres.lookup_dbdir('WS_ALL')
    # self = ingest_dataset.Ingestable2(dbdir)

    if False:
        # Show overlap matrix
        import wbia.plottool as pt
        import pandas as pd
        import numpy as np

        dict_ = overlaps
        s = pd.Series(dict_, index=pd.MultiIndex.from_tuples(overlaps))
        df = s.unstack()
        lhs, rhs = df.align(df.T)
        df = lhs.add(rhs, fill_value=0).fillna(0)

        label_texts = df.columns.values

        def label_ticks(label_texts):
            import wbia.plottool as pt

            truncated_labels = [repr(lbl[0:100]) for lbl in label_texts]
            ax = pt.gca()
            ax.set_xticks(list(range(len(label_texts))))
            ax.set_xticklabels(truncated_labels)
            [lbl.set_rotation(-55) for lbl in ax.get_xticklabels()]
            [lbl.set_horizontalalignment('left') for lbl in ax.get_xticklabels()]

            # xgrid, ygrid = np.meshgrid(range(len(label_texts)), range(len(label_texts)))
            # pt.plot_surface3d(xgrid, ygrid, disjoint_mat)
            ax.set_yticks(list(range(len(label_texts))))
            ax.set_yticklabels(truncated_labels)
            [lbl.set_horizontalalignment('right') for lbl in ax.get_yticklabels()]
            [lbl.set_verticalalignment('center') for lbl in ax.get_yticklabels()]
            # [lbl.set_rotation(20) for lbl in ax.get_yticklabels()]

        # df = df.sort(axis=0)
        # df = df.sort(axis=1)

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

    # overlap_df = pd.DataFrame.from_dict(overlap_img_list)

    class TmpImage(ut.NiceRepr):
        pass

    from skimage.feature import hog
    from skimage import data, color, exposure
    import wbia.plottool as pt

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
        fd, hog_image = hog(
            image,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualise=True,
        )

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

    # for


def detect_sharks(ibs, gids):
    # import wbia
    # ibs = wbia.opendb('WS_ALL')
    config = {
        'algo': 'yolo',
        'sensitivity': 0.2,
        'config_filepath': ut.truepath(
            '~/work/WS_ALL/localizer_backup/detect.yolo.2.cfg'
        ),
        'weight_filepath': ut.truepath(
            '~/work/WS_ALL/localizer_backup/detect.yolo.2.39000.weights'
        ),
        'class_filepath': ut.truepath(
            '~/work/WS_ALL/localizer_backup/detect.yolo.2.cfg.classes'
        ),
    }
    depc = ibs.depc_image

    # imgsets = ibs.imagesets(text='Injured Sharks')
    # images = ibs.images(imgsets.gids[0])
    images = ibs.images(gids)
    images = images.compress([ext not in ['.gif'] for ext in images.exts])
    gid_list = images.gids

    # result is a tuple:
    # (score, bbox_list, theta_list, conf_list, class_list)
    results_list = depc.get_property('localizations', gid_list, None, config=config)

    results_list2 = []
    multi_gids = []
    failed_gids = []

    # ibs.set_image_imagesettext(failed_gids, ['Fixme'] * len(failed_gids))
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
            res2 = (gid, bbox_list[idx : idx + 1], theta_list[idx : idx + 1])
            results_list2.append(res2)

    ut.dict_hist(([t[1].shape[0] for t in results_list]))

    localized_imgs = ibs.images(ut.take_column(results_list2, 0))
    assert all([len(a) == 1 for a in localized_imgs.aids])
    old_annots = ibs.annots(ut.flatten(localized_imgs.aids))
    # old_tags = old_annots.case_tags

    # Override old bboxes
    import numpy as np

    bboxes = np.array(ut.take_column(results_list2, 1))[:, 0, :]
    ibs.set_annot_bboxes(old_annots.aids, bboxes)

    if False:
        import wbia.plottool as pt

        pt.qt4ensure()

        inter = pt.MultiImageInteraction(
            ibs.get_image_paths(ut.take_column(results_list2, 0)),
            bboxes_list=ut.take_column(results_list2, 1),
        )
        inter.dump_to_disk('shark_loc', num=50, prefix='shark_loc')
        inter.start()

        inter = pt.MultiImageInteraction(ibs.get_image_paths(failed_gids))
        inter.start()

        inter = pt.MultiImageInteraction(ibs.get_image_paths(multi_gids))
        inter.start()


def train_part_detector():
    """
    Problem:
        healthy sharks usually have a mostly whole body shot
        injured sharks usually have a close up shot.
        This distribution of images is likely what the injur-shark net is picking up on.

    The goal is to train a detector that looks for things that look
    like the distribution of injured sharks.

    We will run this on healthy sharks to find the parts of
    """
    import wbia

    ibs = wbia.opendb('WS_ALL')
    imgset = ibs.imagesets(text='Injured Sharks')
    injured_annots = imgset.annots[0]  # NOQA

    # config = {
    #    'dim_size': (224, 224),
    #    'resize_dim': 'wh'
    # }

    from pydarknet import Darknet_YOLO_Detector

    data_path = ibs.export_to_xml()
    output_path = join(ibs.get_cachedir(), 'training', 'localizer')
    ut.ensuredir(output_path)
    dark = Darknet_YOLO_Detector()
    results = dark.train(data_path, output_path)
    del dark

    localizer_weight_path, localizer_config_path, localizer_class_path = results
    classifier_model_path = ibs.classifier_train()
    labeler_model_path = ibs.labeler_train()
    output_path = join(ibs.get_cachedir(), 'training', 'detector')
    ut.ensuredir(output_path)
    ut.copy(localizer_weight_path, join(output_path, 'localizer.weights'))
    ut.copy(localizer_config_path, join(output_path, 'localizer.config'))
    ut.copy(localizer_class_path, join(output_path, 'localizer.classes'))
    ut.copy(classifier_model_path, join(output_path, 'classifier.npy'))
    ut.copy(labeler_model_path, join(output_path, 'labeler.npy'))

    # ibs.detector_train()


def purge_ensure_one_annot_per_images(ibs):
    """
    pip install Pipe
    """
    # Purge all but one annotation
    images = ibs.images()
    # images.aids
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
    # list(map(basename, map(dirname, images.uris_original)))

    def VecPipe(func):
        import pipe

        @pipe.Pipe
        def wrapped(sequence):
            return map(func, sequence)
            # return (None if item is None else func(item) for item in sequence)

        return wrapped

    name_list = list(images.uris_original | VecPipe(dirname) | VecPipe(basename))
    aids_list = images.aids
    ut.assert_all_eq(list(aids_list | VecPipe(len)))
    annots = ibs.annots(ut.flatten(aids_list))
    annots.names = name_list


def shark_misc():
    import wbia

    ibs = wbia.opendb('WS_ALL')
    aid_list = ibs.get_valid_aids()
    flag_list = ibs.get_annot_been_adjusted(aid_list)
    adjusted_aids = ut.compress(aid_list, flag_list)
    return adjusted_aids

    # if False:
    #    # TRY TO FIGURE OUT WHY URLS ARE MISSING IN STEP 1
    #    encounter_to_parsed1 = parsed1.group_items('encounter')
    #    encounter_to_parsed2 = parsed2.group_items('encounter')

    #    url_to_parsed1 = parsed1.group_items('img_url')
    #    url_to_parsed2 = parsed2.group_items('img_url')

    #    def set_overlap(set1, set2):
    #        set1 = set(set1)
    #        set2 = set(set2)
    #        return ut.odict([
    #            ('s1', len(set1)),
    #            ('s2', len(set2)),
    #            ('isect', len(set1.intersection(set2))),
    #            ('union', len(set1.union(set2))),
    #            ('s1 - s2', len(set1.difference(set2))),
    #            ('s2 - s1', len(set2.difference(set1))),
    #        ])
    #    print('encounter overlap: ' + ut.repr3(set_overlap(encounter_to_parsed1, encounter_to_parsed2)))
    #    print('url overlap: ' + ut.repr3(set_overlap(url_to_parsed1, url_to_parsed2)))

    #    url1 = list(url_to_parsed1.keys())
    #    url2 = list(url_to_parsed2.keys())
    #    # remove common prefixes
    #    from os.path import commonprefix, basename  # NOQA
    #    cp1 = commonprefix(url1)
    #    cp2 = commonprefix(url2)
    #    #suffix1 = sorted([u[len(cp1):].lower() for u in url1])
    #    #suffix2 = sorted([u[len(cp2):].lower() for u in url2])
    #    suffix1 = sorted([u[len(cp1):] for u in url1])
    #    suffix2 = sorted([u[len(cp2):] for u in url2])
    #    print('suffix overlap: ' + ut.repr3(set_overlap(suffix1, suffix2)))
    #    set1 = set(suffix1)
    #    set2 = set(suffix2)
    #    only1 = list(set1 - set1.intersection(set2))
    #    only2 = list(set2 - set1.intersection(set2))

    #    import numpy as np
    #    for suf in ut.ProgIter(only2, bs=True):
    #        dist = np.array(ut.edit_distance(suf, only1))
    #        idx = ut.argsort(dist)[0:3]
    #        if dist[idx][0] < 3:
    #            close = ut.take(only1, idx)
    #            print('---')
    #            print('suf = %r' % (join(cp2, suf),))
    #            print('close = %s' % (ut.repr3([join(cp1, c) for c in close]),))
    #            print('---')
    #            break

    #    # Associate keywords with original images
    #    #lower_urls = [x.lower() for x in parsed['img_url']]
    #    url_to_idx = ut.make_index_lookup(parsed1['img_url'])
    #    parsed1['keywords'] = [[] for _ in range(len(parsed1))]
    #    for url, keys in url_to_keys.items():
    #        # hack because urls are note in the same format
    #        url = url.replace('wildbook_data_dir', 'shepherd_data_dir')
    #        url = url.lower()
    #        if url in url_to_idx:
    #            idx = url_to_idx[url]
    #            parsed1['keywords'][idx].extend(keys)

    # healthy_annots = ibs.annots(ibs.imagesets(text='Non-Injured Sharks').aids[0])
    # ibs.set_annot_prop('healthy', healthy_annots.aids, [True] * len(healthy_annots))
    # ['healthy' in t and len(t) > 0 for t in single_annots.case_tags]
    # healthy_tags = []

    # ut.find_duplicate_items(cur_img_uuids)
    # ut.find_duplicate_items(new_img_uuids)
    # cur_uuids = set(cur_img_uuids)
    # new_uuids = set(new_img_uuids)
    # both_uuids = new_uuids.intersection(cur_uuids)
    # only_cur = cur_uuids - both_uuids
    # only_new = new_uuids - both_uuids
    # print('len(cur_uuids) = %r' % (len(cur_uuids)))
    # print('len(new_uuids) = %r' % (len(new_uuids)))
    # print('len(both_uuids) = %r' % (len(both_uuids)))
    # print('len(only_cur) = %r' % (len(only_cur)))
    # print('len(only_new) = %r' % (len(only_new)))

    # Ensure that data in both sets are syncronized
    # images_both = []

    # if False:
    #    print('Removing small images')
    #    import numpy as np
    #    import vtool as vt
    #    imgsize_list = np.array([vt.open_image_size(gpath) for gpath in parsed['new_fpath']])
    #    sqrt_area_list = np.sqrt(np.prod(imgsize_list, axis=1))
    #    areq_flags_list = sqrt_area_list >= 750
    #    parsed = parsed.compress(areq_flags_list)
