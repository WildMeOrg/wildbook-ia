#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
#import getopt
#import sys
from os.path import splitext, join, exists, dirname
from os.path import commonprefix, basename  # NOQA
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[getshark]')


def sync_whalesharks():
    """
    MAIN ENTRY POINT

    Syncronizes our ibeis database with whaleshark.org

    #cd ~/work/WS_ALL
    python -m ibeis.scripts.getshark

    cd /media/raid/raw/WhaleSharks_WB/

    >>> from ibeis.scripts.getshark import *  # NOQA
    """
    from ibeis.scripts import getshark

    # Prepare the output directory for writing, if it doesn't exist
    output_dir = 'sharkimages'
    ut.ensuredir(output_dir)
    download_dir = join('/media/raid/raw/WhaleSharks_WB/', output_dir)

    # Read ALL data from whaleshark.org
    parsed = getshark.parse_whaleshark_org()
    parsed = postprocess_filenames(parsed, download_dir)
    parsed = postprocess_tags(parsed)

    # Download images that we dont have yet
    download_missing_images(parsed)

    # Change variable name to info now that we downloaded
    info = parsed.copy()
    del info['ext']
    del info['localid']

    # Remove corrupted or ill-formatted images
    print('Checking for corrupted images')
    import vtool as vt
    isvalid = vt.filterflags_valid_images(info['new_fpath'])
    info = info.compress(isvalid)

    # Rectify duplicate information
    # Stride of 1 is what IA uses internally
    _prog = ut.ProgPartial(bs=True, freq=10)
    info['uuid'] = [ut.get_file_uuid(fpath_, stride=1)
                    for fpath_ in _prog(info['new_fpath'], lbl='uuid check')]

    info_groups = info.group(info['uuid'])[1]
    multi_groups  = [g for g in info_groups if len(g) > 1]
    single_groups = [g for g in info_groups if len(g) == 1]

    fixed_groups = []
    for group in multi_groups:
        newgroup = {}
        for key in group.keys():
            val = group[key]
            # flatten tag lists otherwise take the first item
            if isinstance(val[0], (tuple, list)):
                val_ = ut.unique(ut.flatten(val))
            else:
                val_ = val[0]
            newgroup[key] = [val_]
        fixed_groups.append(ut.ColumnLists(newgroup))
    singles = ut.ColumnLists.flatten(single_groups)
    fixed = ut.ColumnLists.flatten(fixed_groups)
    info = singles + fixed

    # Check these images against what currently exists in WS_ALL
    import ibeis
    import numpy as np
    ibs = ibeis.opendb('WS_ALL')
    #images = ibs.images()
    #cur_img_uuids = [ut.get_file_uuid(fpath_, stride=8)
    #                 for fpath_ in _prog(images.paths)]
    #cur_img_uuids = [ut.get_file_uuid(fpath_, stride=1)
    #                 for fpath_ in _prog(images.paths, freq=10)]
    #new_img_uuids = info['uuid']

    # Get info for items not yet in database
    if True:
        info_gid_list = ibs.get_image_gids_from_uuid(info['uuid'])
        new_info = info.compress(ut.flag_None_items(info_gid_list))

        gid_list = ibs.add_images(new_info['new_fpath'])
        new_info['gid'] = gid_list

        failed_flags = ut.flag_None_items(new_info['gid'])
        print('# failed %s' % (sum(failed_flags)),)
        passed_flags = ut.not_list(failed_flags)
        new_info = new_info.compress(passed_flags)
        ut.assert_all_not_None(new_info['gid'])
        #ibs.get_image_uris_original(clist['gid'])
        assert len(ut.find_duplicate_items(new_info['gid'])) == 0
        ibs.set_image_uris_original(new_info['gid'], new_info['img_url'], overwrite=True)

        images_new = ibs.images(new_info['gid'])
        is_empty_annots = np.array(images_new.num_annotations) == 0

        # TODO: multis
        empty_new_info = new_info.compress(is_empty_annots)
        empty_new_images = images_new.compress(is_empty_annots)
        gids = empty_new_images.gids
        #empty_cur_annots = ibs.annots(ut.flatten(empty_cur_images.aids))

        config = {
            'algo'            : 'yolo',
            'sensitivity'     : 0.2,
            'config_filepath' : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.cfg'),
            'weight_filepath' : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.39000.weights'),
            'class_filepath'  : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.cfg.classes'),
        }
        depc = ibs.depc_image

        images = ibs.images(gids)
        images = images.compress([ext_ not in ['.gif'] for ext_ in images.exts])
        gid_list = images.gids

        # result is a tuple: (score, bbox_list, theta_list, conf_list, class_list)
        results_list = depc.get_property('localizations', gid_list, None, config=config)  # NOQA

        results_list2 = []
        multi_gids = []
        failed_gids = []

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

        if False:
            ibs.set_image_imagesettext(failed_gids, ['Fixme'] * len(failed_gids))
            ibs.set_image_imagesettext(multi_gids, ['Fixme2'] * len(multi_gids))

        # Reorder empty_info to be aligned with results
        localized_imgs = ibs.images(ut.take_column(results_list2, 0))
        empty_new_info_ = empty_new_info.loc_by_key('gid', localized_imgs.gids)
        assert all([len(a) == 0 for a in localized_imgs.aids])

        #old_annots = ibs.annots(ut.flatten(localized_imgs.aids))
        #old_tags = old_annots.case_tags

        # Override old bboxes
        bboxes = np.array(ut.take_column(results_list2, 1))[:, 0, :]
        thetas = np.array(ut.take_column(results_list2, 2))[:, 0]
        names = empty_new_info_['nameid']
        species = ['whale_shark'] * len(localized_imgs)
        aid_list = ibs.add_annots(localized_imgs.gids, bbox_list=bboxes,
                                  theta_list=thetas, name_list=names,
                                  species_list=species)
        aid_list
        #ibs.set_annot_bboxes(old_annots.aids, bboxes)

    if True:
        info_gid_list = ibs.get_image_gids_from_uuid(info['uuid'])
        # Get info for items already in database
        cur_info = info.compress(ut.flag_not_None_items(info_gid_list))
        cur_info['gid'] = ut.compress(info_gid_list, ut.flag_not_None_items(info_gid_list))
        cur_images = ibs.images(cur_info['gid'])
        assert cur_images.uuids == cur_info['uuid']
        num_bad_orig_uri = sum([x != y for x, y in zip(cur_images.uris_original, cur_info['img_url'])])
        sum([x == y for x, y in zip(cur_images.gnames, cur_info['orig_fname'])])

        # Use the wildbook urls as original uris
        if num_bad_orig_uri > 0:
            print('Fixing %d original uris' % (num_bad_orig_uri,))
            ibs.set_image_uris_original(cur_images.gids, cur_info['img_url'], overwrite=True)
        else:
            print('All %d original uris do not need fixing' % len(cur_images))

        # TODO: multis
        is_single_annots = np.array(cur_images.num_annotations) == 1
        single_cur_info = cur_info.compress(is_single_annots)
        single_cur_images = cur_images.compress(is_single_annots)
        single_cur_annots = ibs.annots(ut.flatten(single_cur_images.aids))

        # Map viewpoint onto images
        mapping = [
            ('view-left', 'left'),
            ('view-right', 'right'),
            ('view-back', 'back'),
        ]
        for tag, yaw_text in mapping:
            tag_flags = ut.filterflags_general_tags(single_cur_info['tags'], has_any=[tag])
            _annots = single_cur_annots.compress(tag_flags)
            _annots = _annots.compress(ut.flag_None_items(_annots.yaw_texts))
            print('%d/%d annots need fixed yaws %r' % (len(_annots), len(single_cur_annots), tag,))
            _annots.yaw_texts = [yaw_text] * len(_annots)

        # Fix Species
        bad_flags = [s == '____' for s in single_cur_annots.species]
        _annots = single_cur_annots.compress(bad_flags)
        print('%d/%d annots need fixed species' % (len(_annots), len(single_cur_annots)))
        _annots.species = ['whale_shark'] * len(_annots)

        # Fix Injur / Healthy Tags
        flags = (np.array(ut.lmap(len, single_cur_annots.case_tags)) == 0)
        _annots = single_cur_annots.compress(flags)
        _info = single_cur_info.compress(flags)
        print('%d/%d annots have no tags' % (len(_annots), len(single_cur_annots)))
        # reparse tags (dev only, delete the next line)
        _info['tags'] = getshark.parse_shark_fname_tags(_info['orig_fname'])
        print('Tags from WB imgnames:' +
              ut.repr3(ut.dict_hist(ut.flatten(_info['tags']))))
        probably_healthy_flags = ut.filterflags_general_tags(_info['tags'],
                                                             none_match=['injur-.*', 'cropped', 'notch'])
        probably_healthy_annots = _annots.compress(probably_healthy_flags)
        ibs.set_annot_prop('healthy', probably_healthy_annots.aids, [True] * len(probably_healthy_annots))
        ibs.set_image_imagesettext(probably_healthy_annots.gids, ['Probably Healthy'] * len(probably_healthy_annots))

        probably_injured_flags = ut.filterflags_general_tags(_info['tags'], any_startswith=('injur-'))
        probably_injured_annots = _annots.compress(probably_injured_flags)
        injur_tags = [[t for t in ts if t.startswith('injur-')] for ts in _info['tags']]
        injur_tags = ut.compress(injur_tags, probably_injured_flags)
        probably_injured_annots = _annots.compress(probably_injured_annots)
        ibs.append_annot_case_tags(probably_injured_annots.aids, injur_tags)
        ibs.set_image_imagesettext(probably_injured_annots.gids, ['Probably Injured'] * len(probably_injured_annots))

        # manually reviewed some
        flags = ut.filterflags_general_tags(probably_healthy_annots.case_tags, has_any=['error:other'])
        actually_unhealthy = probably_healthy_annots.compress(flags)
        ibs.append_annot_case_tags(actually_unhealthy.aids, ['injur-unknown'] * len(actually_unhealthy))
        ibs.set_annot_prop('healthy', actually_unhealthy.aids, [False] * len(actually_unhealthy))
        ibs.set_annot_prop('error:other', actually_unhealthy.aids, [False] * len(actually_unhealthy))
        ibs.set_image_imagesettext(actually_unhealthy.gids, ['Probably Injured'] * len(actually_unhealthy))

        ibs.unrelate_images_and_imagesets(actually_unhealthy.gids, [ibs.imagesets(text='Probably Healthy')._rowids[0]] * len(actually_unhealthy.gids))

        # TODO: cls

        x = ibs.annots(ibs.imagesets(text='Non-Injured Sharks').aids[0])
        y = ibs.annots(ibs.imagesets(text='Injured Sharks').aids[0])

        #healthy_annots = ibs.annots(ibs.imagesets(text='Non-Injured Sharks').aids[0])
        #ibs.set_annot_prop('healthy', healthy_annots.aids, [True] * len(healthy_annots))
        #['healthy' in t and len(t) > 0 for t in single_cur_annots.case_tags]
        #healthy_tags = []

    #ut.find_duplicate_items(cur_img_uuids)
    #ut.find_duplicate_items(new_img_uuids)
    #cur_uuids = set(cur_img_uuids)
    #new_uuids = set(new_img_uuids)
    #both_uuids = new_uuids.intersection(cur_uuids)
    #only_cur = cur_uuids - both_uuids
    #only_new = new_uuids - both_uuids
    #print('len(cur_uuids) = %r' % (len(cur_uuids)))
    #print('len(new_uuids) = %r' % (len(new_uuids)))
    #print('len(both_uuids) = %r' % (len(both_uuids)))
    #print('len(only_cur) = %r' % (len(only_cur)))
    #print('len(only_new) = %r' % (len(only_new)))

    # Ensure that data in both sets are syncronized
    #images_both = []

    if False:
        print('Removing small images')
        import numpy as np
        imgsize_list = np.array([vt.open_image_size(gpath) for gpath in parsed['new_fpath']])
        sqrt_area_list = np.sqrt(np.prod(imgsize_list, axis=1))
        areq_flags_list = sqrt_area_list >= 750
        parsed = parsed.compress(areq_flags_list)

    if False:
        grouped_idxs = ut.group_items(list(range(len(parsed['nameid']))),
                                      parsed['nameid'])
        keep_idxs = sorted(ut.flatten([idxs for key, idxs in grouped_idxs.items() if len(idxs) >= 2]))
        parsed = parsed.take(keep_idxs)

    if False:
        print('Moving images to secondary directory')
        named_outputdir = 'named-left-sharkimages'
        # Build names
        parsed['namedir_fpath'] = [
            join(named_outputdir, _nameid, _fname)
            for _fname, _nameid in zip(parsed['new_fname'],
                                       parsed['nameid'])]
        # Create directories
        ut.ensuredir(named_outputdir)
        named_dirs = ut.unique_ordered(list(map(dirname, parsed['namedir_fpath'])))
        for dir_ in named_dirs:
            ut.ensuredir(dir_)
        # Copy
        ut.copy_files_to(src_fpath_list=parsed['new_fpath'],
                         dst_fpath_list=parsed['namedir_fpath'])


def _needs_redownload(fpath, seconds_thresh):
    if exists(fpath):
        file_info = ut.get_file_info(fpath)
        dt = ut.parse_timestamp(file_info['last_modified'], utc=True)
        delta = dt - ut.utcnow_tz()
        redownload = delta.total_seconds() > seconds_thresh
    else:
        redownload = True
    return redownload


def parse_whaleshark_org():
    """
    Read list of all iamges from wildbook

    >>> from ibeis.scripts.getshark import *  # NOQA
    """
    from xml.dom.minidom import parseString
    from ibeis.scripts import getshark

    url = 'www.whaleshark.org/listImages.jsp'
    number = None

    cache_dpath = ut.ensure_app_resource_dir('utool', 'sharkinfo')
    cache_fapth = join(cache_dpath, 'listImagesSharks.xml')

    # redownload every 30 days or so
    if getshark._needs_redownload(cache_fapth, 60 * 60 * 24 * 30):
        XMLdata = ut.url_read_text(url)
        ut.writeto(cache_fapth, XMLdata)
    else:
        XMLdata = ut.readfrom(cache_fapth)

    # Parse attributes out of XML
    dom = parseString(XMLdata.encode('utf8'))
    if number:
        maxCount = min(number, len(dom.getElementsByTagName('img')))
    else:
        maxCount = len(dom.getElementsByTagName('img'))
    parsed_info = ut.ddict(list)
    print('Reading XML information from %d images...' % maxCount)
    shark_elements = dom.getElementsByTagName('shark')
    _prog = ut.ProgPartial(bs=True, freq=10)
    for shark in _prog(shark_elements, lbl='parsing shark elements'):
        localCount = 0
        for encounter in shark.getElementsByTagName('encounter'):
            for img in encounter.getElementsByTagName('img'):
                localCount += 1

                img_url = img.getAttribute('href')
                ext = splitext(img_url)[1].lower()
                nameid = shark.getAttribute('number')

                new_fname = '%s-%i%s' % (
                    nameid, localCount, ext)

                parsed_info['img_url'].append(img_url)
                parsed_info['nameid'].append(nameid)
                parsed_info['localid'].append(localCount)
                # might be different due to prefix
                parsed_info['new_fname'].append(new_fname)

                parsed_info['encounter'].append(encounter.getAttribute('number'))

                #print('Parsed %i / %i files.' % (len(parsed_info['orig_fname']), maxCount))
                if number is not None and len(parsed_info['orig_fname']) == number:
                    break

    keywords, url_to_keys, parsed2 = getshark.parse_whaleshark_org_keywords()
    print('Parsed %d keywords' % (len(keywords),))
    print('Parsed %d urls from keywords' % (len(parsed2),))

    parsed_ = ut.ColumnLists(parsed_info)
    print('Parsed %d urls' % (len(parsed_),))

    # Check and rectify for duplicate urls
    unique_urls, idxs = parsed_.group_indicies('img_url')
    toremove = []
    for url, idxs in zip(unique_urls, idxs):
        if len(idxs) == 1:
            continue
        dupinfo = parsed_.take(idxs)
        del dupinfo[['localid', 'new_fname', 'img_url']]
        can_fix = True
        for key, vals in dupinfo.asdict().items():
            if not ut.allsame(vals):
                print(dupinfo.ascsv())
                print(('Duplicate items have different values'))
                # May need to fix a case when annoations happen in WB
                assert False, 'cant have this happen'
                can_fix = False
        if can_fix:
            toremove += idxs[1:]
    print('Removing %d duplicate urls' % (len(toremove),))
    flags = ut.not_list(ut.index_to_boolmask(toremove))
    parsed1 = parsed_.compress(flags)

    prefix = commonprefix(parsed1['img_url'])
    parsed1['suffix'] = [url_[len(prefix):] for url_ in parsed1['img_url']]

    # Apply keywords to existing images
    suffix_to_idx = ut.make_index_lookup(parsed1['suffix'])
    matching_idx = ut.dict_take(suffix_to_idx, parsed2['suffix'], None)
    match_idx1 = ut.filter_Nones(matching_idx)
    #matching1 = parsed1.take(match_idx1)
    matching2 = parsed2.take(ut.where(matching_idx))
    print('There are %d items in common' % (len(matching2),))
    nonmatching2 = parsed2.take(ut.where(ut.not_list(matching_idx)))

    # add in default values for parsed2
    nonmatching2['nameid'] = [None] * len(nonmatching2)
    nonmatching2['localid'] = [None] * len(nonmatching2)

    # Merge keywords from matching parts in parsed2 into parsed1
    parsed1['keywords'] = [[] for _ in range(len(parsed1))]
    for idx1, keys in zip(match_idx1, matching2['keywords']):
        parsed1['keywords'][idx1].extend(keys)

    parsed = parsed1 + nonmatching2
    print('Parsed %d total urls' % (len(parsed),))
    return parsed

    #if False:
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


def parse_whaleshark_org_keywords():
    from ibeis.scripts import getshark
    url = 'http://www.whaleshark.org/getKeywordImages.jsp'
    cache_dpath = ut.ensure_app_resource_dir('utool', 'sharkinfo')

    def cached_json_request(url_):
        import requests
        cache_fpath = join(cache_dpath, 'req_' + ut.hashstr27(url_) + '.json')
        if getshark._needs_redownload(cache_fpath, 60 * 60 * 24 * 30):
            resp = requests.get(url_)
            assert resp.status_code == 200
            dict_ = resp.json()
            ut.save_data(cache_fpath, dict_)
        else:
            dict_ = ut.load_data(cache_fpath)
        return dict_

    keywords = cached_json_request(url)['keywords']
    key_list = ut.take_column(keywords, 'indexName')

    keyed_images = {}
    for key in ut.ProgIter(key_list, lbl='reading index', bs=True):
        key_url = url + '?indexName={indexName}'.format(indexName=key)
        keyed_images[key] = cached_json_request(key_url)['images']

    url_to_keys = ut.ddict(list)
    for key, images in keyed_images.items():
        for imgdict in images:
            url_to_keys[imgdict['url']].append(key)

    parsed_info2 = ut.ddict(list)
    for key, images in keyed_images.items():
        for imgdict in images:
            parsed_info2['img_url'].append(imgdict['url'])
            parsed_info2['encounter'].append(imgdict['correspondingEncounterNumber'])
            parsed_info2['keywords'].append([key])
    parsed2_ = ut.ColumnLists(parsed_info2)

    # Assert no unfixable duplicates exist
    dups = ut.find_duplicate_items(parsed2_['img_url'])
    for url, idxs in dups.items():
        dupinfo = parsed2_.take(idxs)
        del dupinfo[['img_url', 'keywords']]
        for key, vals in dupinfo.asdict().items():
            if not ut.allsame(vals):
                print(dupinfo.ascsv())
                print(('Duplicate items have different values'))
                # May need to fix a case when annoations happen in WB
                assert False, 'cant have this happen'

    # Rectiry expected duplicate info
    groups = parsed2_.group(parsed2_['img_url'])[1]
    d = {}
    d['keywords'] = [ut.unique(ut.flatten(g['keywords'])) for g in groups]
    for key in parsed2_.keys():
        if key == 'keywords':
            continue
        vals = [g[key] for g in groups]
        assert all([ut.allsame(v) for v in vals])
        d[key] = ut.take_column(vals, 0)
    parsed2 = ut.ColumnLists(d)

    prefix = commonprefix(parsed2['img_url'])
    parsed2['suffix'] = [url_[len(prefix):] for url_ in parsed2['img_url']]
    parsed2['new_fname'] = [suffix.replace('/', '--') for suffix in parsed2['suffix']]
    return keywords, url_to_keys, parsed2


def postprocess_filenames(parsed, download_dir):
    from os.path import commonprefix, basename  # NOQA
    # Postprocess
    parsed['new_fpath'] = [join(download_dir, _fname)
                                for _fname in parsed['new_fname']]
    prefix = commonprefix(parsed['img_url'])
    parsed['orig_fname'] = [url_[len(prefix):] for url_ in parsed['img_url']]

    parsed['ext'] = [splitext(_fname)[-1] for _fname in parsed['new_fname']]

    # Filter based on image type (keep only jpgs)
    ext_flags = [ext_ in ['.jpg', '.jpeg'] for ext_ in parsed['ext']]

    parsed = parsed.compress(ext_flags)
    num_removed = sum(ut.not_list(ext_flags))
    print('Removed %d images based on extensions' % (num_removed,))
    return parsed


def postprocess_tags(parsed):
    # Filter to only images matching the appropriate tags
    from ibeis.scripts import getshark
    parsed['tags'] = getshark.parse_shark_fname_tags(parsed['orig_fname'])
    # add keywords into tags
    for t, k in zip(parsed['tags'], parsed['keywords']):
        t += k

    # Map tags
    tags_list = parsed['tags']
    cleaned_tags = ut.clean_tags(
        tags_list,
        direct_map=[
            ('c429b13e4d232129014d251c74c60011', 'stranding'),
            ('', None),
        ],
        regex_aug=[
            ('other_injury', 'injur-other'),
            ('truncation', 'injur-trunc'),
            ('nicks', 'injur-nicks'),
            ('scar', 'injur-scar'),
            ('bite', 'injur-bite'),
        ],
    )
    cleaned_tags = ut.clean_tags(
        cleaned_tags,
        regex_aug=[
            ('injur-', 'injured'),
        ],
    )
    parsed['tags'] = cleaned_tags

    tag_flags = ut.filterflags_general_tags(
        parsed['tags'],
        #has_any=['view-left'],
        #none_match=['qual.*', 'view-top', 'part-.*', 'cropped'],
    )
    if all(tag_flags):
        print('Tags histogram:' +
              ut.repr3(ut.dict_hist(ut.flatten(parsed['tags']), ordered=True)))
    else:
        print('Tags before choosing:' +
              ut.repr3(ut.dict_hist(ut.flatten(parsed['tags']))))
        parsed = parsed.compress(tag_flags)
        print('Tags after choosing:' +
              ut.repr3(ut.dict_hist(ut.flatten(parsed['tags']))))
    num_removed = sum(ut.not_list(tag_flags))
    print('Removed %d images based on tags' % (num_removed,))
    return parsed


def download_missing_images(parsed):
    exist_flags = ut.lmap(exists, parsed['new_fpath'])
    missing_flags = ut.not_list(exist_flags)
    print('nExist = %r / %r' % (sum(exist_flags), len(exist_flags)))
    print('nMissing = %r / %r' % (sum(missing_flags), len(exist_flags)))
    if any(missing_flags):
        missing = parsed.compress(missing_flags)
        print('Downloading missing subset')
        _iter = list(zip(missing['img_url'], missing['new_fpath']))
        _prog = ut.ProgPartial(bs=True, freq=10)
        for img_url, new_fpath in _prog(_iter, lbl='downloading sharks'):
            try:
                ut.download_url(img_url, new_fpath, verbose=False)
            except ZeroDivisionError:
                pass


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
    if False:
        # Can run detection instead
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


def detect_sharks(ibs, gids):
    #import ibeis
    #ibs = ibeis.opendb('WS_ALL')
    config = {
        'algo'            : 'yolo',
        'sensitivity'     : 0.2,
        'config_filepath' : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.cfg'),
        'weight_filepath' : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.39000.weights'),
        'class_filepath'  : ut.truepath('~/work/WS_ALL/localizer_backup/detect.yolo.2.cfg.classes'),
    }
    depc = ibs.depc_image

    #imgsets = ibs.imagesets(text='Injured Sharks')
    #images = ibs.images(imgsets.gids[0])
    images = ibs.images(gids)
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
    import ibeis
    ibs = ibeis.opendb('WS_ALL')
    imgset = ibs.imagesets(text='Injured Sharks')
    injured_annots = imgset.annots[0]  # NOQA

    #config = {
    #    'dim_size': (224, 224),
    #    'resize_dim': 'wh'
    #}

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
    ut.copy(localizer_class_path,  join(output_path, 'localizer.classes'))
    ut.copy(classifier_model_path, join(output_path, 'classifier.npy'))
    ut.copy(labeler_model_path,    join(output_path, 'labeler.npy'))

    # ibs.detector_train()


def parse_shark_fname_tags(orig_fname_list):
    """
    Parses potential tags from the filename

    >>> orig_fname_list = parsed['orig_fname']
    """
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
        'a', 'b', 'c', 's', 'd', 'h', 'g'
        'gender', 'sex',
        'img', 'image', 'pic', 'pics', 'leith', 'trips', 'kings', 'photo', 'video', 'media',
        'fix', 'feeding',
        'nrd', 'nd', 'gen', 'wa', 'nmp', 'bo', 'kd', 'ow', 'ne', 'dsc', 'nwd',
        'mg', 'w', 'mai', 'blue', 'stumpy',
        'oea', 'cbe', 'edc', 'knrt',
        'tiws2',
        'ando', 'adv', 'str', 'adventure',
        'camera', 'tag', 'id', 'ws1', 'ws',
        'gulf', 'wally', 'walhai', 'wags', 'shark[0-9][a-z]',
        'shark', 'sharks', 'reef', '720x480', 'nb', 'nrdive', 'tiws', 'exmouth', 'nrdive2',
        'ningaloo', 'ti', 'nwss', '1st', 'exp', 'wsnd', 'cba', '3iwsd', 'c1',
        'nwd2', 's1l1', 's1r1'
        'encounter',
        'of', 'and', 'the',  'on', 'to', 'with', 'in',
        'up', 'ws3', 's2'
        'tagged', 'from', 'dive', 'untag', 'tagtrace',
        'day', '\\d*april', '\\d*may', '\\d*july', '\\d*june', 'apr\\d+'
        'ningaloo', 'ningblue\\d*', 'kooling',
    ]

    couldbe_tags = [
        'remnant', 'prop', 'north'
        'shot', 'professional', 'red', 'original',
        'measure', 'gender',
        'encounter'
    ]

    invalid_tag_patterns += couldbe_tags

    valid_tag_level_set = [
        ['view-left', 'left', 'lhs', 'l', 'leftside'],
        ['view-right', 'right', 'rhs', 'r', 'rightside'],
        ['view-back', 'back'],
        ['view-top', 'top'],
        ['sex-male', 'male', 'm', 'sexm'],
        ['sex-female', 'female', 'f'],
        ['sex-unknown', 'unknown', 'u'],
        ['part-tail', 'tail', 'caudal'],
        ['part-flank', 'side', 'flank'],
        ['part-head', 'head'],
        ['part-pectoral', 'pectoral', 'pec'],
        ['part-dorsal', 'dorsal', 'dorsals'],
        ['part-claspers', 'claspers', 'clasper'],
        ['part-fin', 'fin', 'leftpecfin'],
        ['part-pelvis', 'pelvic'],
        ['part-gill', 'gill'],
        ['cropped', 'crop'],
        ['injur-scar', 'scar', 'scar2', 'scars', 'tailscar'],
        ['injur-bite', 'bite', 'tailbite'],
        ['injur-nicks', 'scratches', 'nicks', 'headnick'],
        ['injur-damage', 'damage'],
        ['injur-trunc', 'trunc'],
        ['notch'],
        ['small'],
        ['qual-resize', 'resize'],
        ['qual-stretched', 'stretched'],
    ]

    cam_tags = [
        ['cam-slr2', 'slr2'],
        ['cam-5m', '5m'],
        ['cam-7m', '7m'],
        ['cam-4m', '4m'],
        ['copy'],
    ]

    invalid_tag_patterns += [re.escape(c) for c in ut.flatten(cam_tags)]
    #valid_tag_level_set += invalid_tag_patterns

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
        from os.path import basename
        base = basename(splitext(fname)[0])
        #base.replace('(', '')
        #base.replace(')', '')
        _tags = [base]
        for c in ['_', '.', '/', ')', '(', ',']:
            _tags = ut.flatten([t.split(c) for t in _tags])
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

    if 0:
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


#def main():
#    try:
#        opts, args = getopt.getopt(sys.argv[1:], 'f:u:n:h')
#    except getopt.GetoptError:
#        usage()
#        sys.exit(1)

#    filename = None
#    url = 'www.whaleshark.org/listImages.jsp'
#    number = 0

#    # Handle command-line arguments
#    for opt, arg in opts:
#        if opt == '-h':
#            usage()
#            sys.exit()
#        elif opt == '-f':
#            filename = arg
#        elif opt == '-u':
#            url = arg
#        elif opt == '-n':
#            try:
#                number = int(arg)
#            except ValueError:
#                usage()
#                sys.exit()

#    # Open the XML file and extract its contents as a DOM object
#    if filename:
#        XMLdata = ut.readfrom(filename)
#    else:
#        XMLdata = ut.url_read(url)
#        #with open('XMLData.xml', 'w') as file_:
#        #    file_.write(XMLdata)
#    print('Downloading')
#    download_sharks(XMLdata, number)
#    #download_sharks(XMLdata, number)


#def usage():
#    print('Fetches a number of images from the ECOCEAN shark database.')
#    print('Options:')
#    print('  -f <FILENAME> - Reads XML data from a file, rather than a URL.')
#    print('  -u <URL> - Reads XML data from the given URL.')
#    print('  -n <NUMBER> - Number of images to read; if omitted, reads all of them.')
#    print('  -h - Prints this help text.')


#if __name__ == '__main__':
#    main()
