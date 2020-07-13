#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import splitext, join, exists, commonprefix
import utool as ut
import re

(print, rrr, profile) = ut.inject2(__name__, '[getshark]')


def sync_wildbook():
    """
    MAIN ENTRY POINT

    Syncronizes our wbia database with a wildbook database like whaleshark.org

    #cd ~/work/WS_ALL
    python -m wbia.scripts.getshark

    cd /media/raid/raw/WhaleSharks_WB/

    >>> from wbia.scripts.getshark import *  # NOQA
    """
    from wbia.scripts import getshark

    # Prepare the output directory for writing, if it doesn't exist

    if True:
        # Read ALL data from whaleshark.org
        parsed = getshark.parse_whaleshark_org()
        db = 'WS_ALL'
        species = 'whale_shark'
        # images_url = 'http://www.whaleshark.org/listImages.jsp'
        # keyword_url = 'http://www.whaleshark.org/getKeywordImages.jsp'
        download_dir = join('/media/raid/raw/Wildbook/', 'sharkimages')
    if False:
        # Read ALL data from whaleshark.org
        images_url = 'http://www.mantamatcher.org/listImages.jsp'
        keyword_url = None
        db = 'Mantas'
        species = 'manta_ray'
        parsed = parse_wildbook(images_url, keyword_url)
        download_dir = join('/media/raid/raw/Wildbook/', 'mantas')
    if False:
        # Read ALL data from whaleshark.org
        images_url = 'http://www.mantamatcher.org/listMatcherImages.jsp'
        keyword_url = None
        db = 'MantaMatcher'
        species = 'manta_ray'
        parsed = parse_wildbook(images_url, keyword_url)
        download_dir = join('/media/raid/raw/Wildbook/', 'MantaMatcher')

    DRY = False
    DRY = True

    ut.ensuredir(download_dir)
    parsed = getshark.postprocess_filenames(parsed, download_dir)
    parsed = getshark.postprocess_extfilter(parsed)
    parsed = getshark.postprocess_tags_build(parsed)
    parsed = getshark.postprocess_tags_filter(parsed)

    # Download images that we dont have yet
    getshark.download_missing_images(parsed)

    if False:
        parsed._meta['ignore'].extend(['fname_tags', 'tags', 'orig_fname'])

    # Change variable name to info now that we downloaded
    parsed_dl = parsed.copy()
    parsed_dl = getshark.postprocess_corrupted(parsed_dl)
    parsed_dl = getshark.postprocess_uuids(parsed_dl)

    # Name change again after time intensive step
    unmerged = parsed_dl.copy()

    # Squash duplicate images
    info = getshark.postprocess_rectify_duplicates(unmerged)

    # Check these images against what currently exists in WS_ALL
    import wbia

    ibs = wbia.opendb(db, allow_newdir=True)
    all_images = ibs.images()

    num_ia_unique = len(set(all_images.uuids) - set(info['uuid']))

    # TODO: Check that all the UUIDs in the IA database are indeed ok

    # Determine which items are in the database
    info_gid_list = ibs.get_image_gids_from_uuid(info['uuid'])
    is_hit = ut.flag_not_None_items(info_gid_list)
    is_miss = ut.flag_None_items(info_gid_list)
    hit_info = info.compress(is_hit)  # NOQA
    miss_info = info.compress(is_miss)  # NOQA
    print('The IA database has %r images' % (len(all_images),))
    print(
        'The IA database has %r/%r images not from this downloaded set'
        % (num_ia_unique, len(all_images),)
    )

    print('Have %d/%d parsed images' % (len(hit_info), len(info_gid_list)))
    print('Missing %d/%d parsed images' % (len(miss_info), len(info_gid_list)))

    # Add new info
    if not DRY:
        if len(miss_info):
            add_new_images(ibs, miss_info, species)

    # REFIND Existing
    print('Redoing exist check')
    info_gid_list = ibs.get_image_gids_from_uuid(info['uuid'])
    is_hit = ut.flag_not_None_items(info_gid_list)
    is_miss = ut.flag_None_items(info_gid_list)
    hit_info = info.compress(is_hit)  # NOQA
    miss_info = info.compress(is_miss)  # NOQA
    print('The IA database has %r images' % (len(all_images),))
    print(
        'The IA database has %r/%r images not from this downloaded set'
        % (num_ia_unique, len(all_images),)
    )
    print('Have %d/%d parsed images' % (len(hit_info), len(info_gid_list)))
    print('Missing %d/%d parsed images' % (len(miss_info), len(info_gid_list)))

    # Sync existing info
    if True:
        sync_existing_images(ibs, hit_info, species, DRY)


def sync_existing_images(ibs, hit_info, species, DRY):
    print('Syncing existing images')
    import numpy as np

    # Get info for items already in database
    hit_info['gid'] = ibs.get_image_gids_from_uuid(hit_info['uuid'])
    hit_images = ibs.images(hit_info['gid'])

    # Sync original_uris
    print('Checking uris_original')
    ia_prop_list = hit_images.uris_original
    wb_prop_list = hit_info['img_url']
    dirty_flags = []
    for ia_prop, wb_prop in zip(ia_prop_list, wb_prop_list):
        if not ut.is_listlike(ia_prop) and ut.is_listlike(wb_prop):
            # wb had ambigous values, things are ok if we hit at least one
            flag = ia_prop not in wb_prop
        else:
            flag = ia_prop != wb_prop
        dirty_flags.append(flag)
    # Use the wildbook urls as original uris
    if not any(dirty_flags):
        print('...All %d original uris do not need fixing' % len(hit_images))
    else:
        print(
            '...There are %d/%d original uris that need fixing'
            % (sum(dirty_flags), len(dirty_flags))
        )
        dirty_info = hit_info.compress(dirty_flags)
        dirty_gids = dirty_info['gid']
        dirty_wb_props = dirty_info.map_column(
            'img_url', lambda v: ut.ensure_iterable(v)[0]
        )
        if not DRY:
            print('...Fixing %d original uris' % (sum(dirty_flags),))
            ibs.set_image_uris_original(dirty_gids, dirty_wb_props, overwrite=True)
        else:
            print('\n'.join(hit_images.compress(dirty_flags).uris_original))
            dirty_info.print(keys='img_url')

    # Sync info in annotations
    num_annots = np.array(hit_images.num_annotations)

    is_empty = num_annots == 0
    empty_hit_info = hit_info.compress(is_empty)
    empty_hit_images = hit_images.compress(is_empty)

    is_single = num_annots == 1
    single_hit_info = hit_info.compress(is_single)
    single_hit_images = hit_images.compress(is_single)

    is_multi = num_annots > 1
    multi_hit_info = hit_info.compress(is_multi)
    multi_hit_images = hit_images.compress(is_multi)

    print('Syncing annot info in images. Checking annots/per/image')
    print(' * is_empty = %r' % (is_empty.sum(),))
    print(' * is_single = %r' % (is_single.sum(),))
    print(' * is_multi = %r' % (is_multi.sum(),))

    # We exepect an image to be empty if it has junk in the notes
    nonjunk_empty = [n != 'junk' for n in empty_hit_images.notes]
    print('empty_hit_images = %r' % (empty_hit_images.gids,))
    review_empty = empty_hit_images.compress(nonjunk_empty)
    print('need to review %r empty images' % (len(review_empty),))
    if len(review_empty) > 0:
        empty_hit_info.compress(nonjunk_empty).print()
        print('Please manually review empty images %r' % (review_empty.gids,))

    # We expect multi images to be tagged with primary / secondary
    multi_annots = multi_hit_images._annot_groups
    nontagged = [
        not all(ut.filterflags_general_tags(t, has_any=['primary', 'secondary']))
        for t in multi_annots.case_tags
    ]
    if any(nontagged):
        print('Reviewing untagged multi images')
        review_multi = multi_hit_images.compress(nontagged)
        review_annots = review_multi._annot_groups
        print('review_multi = %r' % (review_multi.gids,))
        multi_hit_info.compress(nontagged).print(
            ignore=[
                'img_url',
                'new_fpath',
                'uuid',
                'encounter',
                'keywords',
                'ext',
                'suffix',
                'fname_tags',
                'orig_fname',
                'new_fname',
            ]
        )
        # Determine the primary annotation
        # Check if any are the entire image
        print('Attempting to handle automatically')
        img_areas = np.prod(review_multi.sizes, axis=1)
        bbox_area_rat = [
            np.array(areas) / a for areas, a in zip(review_annots.bbox_area, img_areas)
        ]
        flag = False
        for areas, g in zip(bbox_area_rat, review_multi):
            if np.any(areas > 0.9):
                print('PLEASE CHECK image %r' % (g,))
                flag = True
        assert not flag, 'need to remove bad annots or change code'
        # assert False, 'Need to finish/review this code. Stopped in development'
        primary_idxs = [np.argsort(areas)[-1:] for areas in bbox_area_rat]
        nonprimary_idxs = [np.argsort(areas)[:-1] for areas in bbox_area_rat]
        multi_primary_annots = ibs.annots(
            ut.flatten(ut.ziptake(review_annots.aids, primary_idxs))
        )
        multi_secondary_annots = ibs.annots(
            ut.flatten(ut.ziptake(review_annots.aids, nonprimary_idxs))
        )
        if not DRY:
            multi_primary_annots.append_tags('primary')
            multi_secondary_annots.append_tags('secondary')
            # set tags indicating fg/bg status

    # Take primary annots from multi-images
    isprimary = [
        ut.filterflags_general_tags(t, has_any=['primary'])
        for t in multi_annots.case_tags
    ]
    assert all(p.sum() == 1 for p in isprimary), 'should only be one primary'
    primary_aids = ut.flatten(ut.zipcompress(multi_annots.aids, isprimary))

    # Combine primarys and single_hit into single
    single_annots = ibs.annots(ut.flatten(single_hit_images.aids) + primary_aids)
    single_info = single_hit_info + multi_hit_info

    # Do annot syncing
    sync_annot_info(ibs, single_annots, single_info, species, DRY)


def sync_annot_info(ibs, single_annots, single_info, species, DRY):
    """
    sync `info` from wildbook into `annots` from IA.
    """
    import numpy as np

    # single_info._meta['ignore'] = ['img_url', 'new_fpath', 'uuid', 'encounter']
    single_info._meta['ignore'] = ['img_url', 'new_fpath', 'uuid']

    # Associate single annots and info using aids (lists should correspond)
    single_info['aid'] = single_annots.aids

    if True:
        # try and set encounter information
        key1 = 'encounter'
        prop2 = 'static_encounter'
        repl2 = ('____', None)
        is_set = False
        check_annot_disagree(
            single_info, single_annots, key1, prop2, repl2, is_set, DRY=DRY
        )

    # Fix sharks marked as healthy and injured
    isbadmark = ut.filterflags_general_tags(
        single_annots.case_tags, any_startswith='injur-', has_all='healthy', logic='and'
    )
    print('Marked %r as both healthy and injured' % (isbadmark.sum()))
    if np.any(isbadmark):
        bad_fixme = single_annots.compress(isbadmark)
        if not DRY:
            bad_fixme.remove_tags('healthy')

    # cleaned_tags = ut.modify_tags(
    #    single_info['tags'],
    #    regex_map=[
    #        ('view-.*', None)
    #    ],
    # )

    # info_injur_tags = parse_injury_categories(single_info['tags'])
    # annot_injur_tags = parse_injury_categories(single_annots.case_tags)
    # print(ut.repr4(ut.dict_hist(ut.flatten(cleaned_tags))))

    # info_injur_tags = [t if len(t) > 0 else ['healthy'] for t in info_injur_tags]
    # info_injur_tags = [ut.setdiff(t, ['healthy']) for t in info_injur_tags]
    # annot_injur_tags = [ut.setdiff(t, ['healthy']) for t in annot_injur_tags]
    # info_injur_tags = [ut.setdiff(t, ['injur-other']) for t in info_injur_tags]
    # annot_injur_tags = [ut.setdiff(t, ['injur-other']) for t in annot_injur_tags]

    # Remove redundant aliases on IA side
    cleaned_tags = ut.modify_tags(
        single_annots.case_tags,
        direct_map=[
            ('nicks', 'injur-nicks'),
            ('scar', 'injur-scar'),
            ('trunc', 'injur-trunc'),
            ('injur-trtruunc', 'injur-trunc'),
        ],
    )
    # print(ut.repr4(ut.dict_hist(ut.flatten(cleaned_tags))))
    single_info['orig_case_tags'] = ut.lmap(sorted, single_annots.case_tags)
    single_info['clean_case_tags'] = ut.lmap(sorted, cleaned_tags)

    check_annot_disagree(
        single_info,
        single_annots,
        key1='clean_case_tags',
        prop2=None,
        repl2=(None, []),
        is_set=True,
        key2='orig_case_tags',
        DRY=DRY,
    )
    isdirty = [
        x != y
        for x, y in zip(single_info['orig_case_tags'], single_info['clean_case_tags'])
    ]
    dirty_info = single_info.compress(isdirty)
    print('removing redundant info from %r annots' % (len(dirty_info),))
    if not DRY:
        # dirty_info['orig_case_tags']
        ibs.overwrite_annot_case_tags(dirty_info['aid'], dirty_info['clean_case_tags'])

    # Setup injury tags
    info_injur_tags = get_injured_tags(single_info['tags'])
    # annot_injur_tags = get_injured_tags(single_annots.case_tags)
    single_info['injur_tags'] = info_injur_tags
    # single_info['annot_tags'] = annot_injur_tags
    # # Fix injury tags
    # key1 = 'injur_tags'
    # prop2 = None
    # key2 = 'annot_tags'
    # repl2 = (None, [])
    # is_set = True
    # check_annot_disagree(single_info, single_annots, key1, None, repl2, is_set,
    #                     key2=key2, DRY=DRY)
    # We should just be able to do a union on the two sets.
    isdirty = [
        not ut.issubset(t1, t2)
        for t1, t2 in zip(single_info['injur_tags'], single_annots.case_tags)
    ]
    new_injurtags = [
        ut.setdiff(t1, t2)
        for t1, t2 in zip(single_info['injur_tags'], single_annots.case_tags)
    ]
    single_info['new_injurtags'] = new_injurtags
    dirty_info = single_info.compress(isdirty)
    print('new injur tags' + ut.repr4(ut.dict_hist(ut.flatten(new_injurtags))))
    print('unioning new injur tags into %r/%r annots' % (sum(isdirty), len(isdirty)))
    if not DRY:
        ibs.append_annot_case_tags(dirty_info['aid'], dirty_info['new_injurtags'])

    # Append all other keywords as well
    cleaned_keywords = ut.modify_tags(single_info['keywords'], direct_map=[('', None)])
    single_info['new_keywords'] = [
        ut.setdiff(t1, t2) for t1, t2 in zip(cleaned_keywords, single_annots.case_tags)
    ]
    isdirty = [len(t) > 0 for t in single_info['new_keywords']]
    dirty_info = single_info.compress(isdirty)
    print(
        'new_keywords' + ut.repr4(ut.dict_hist(ut.flatten(single_info['new_keywords'])))
    )
    print('unioning new_keywords into %r/%r annots' % (len(dirty_info), len(isdirty)))
    if not DRY:
        ibs.append_annot_case_tags(dirty_info['aid'], dirty_info['new_keywords'])

    # Check if any other tags need appending
    cleaned_tags = ut.modify_tags(single_info['tags'], direct_map=[('', None)])
    single_info['new_tags'] = [
        ut.setdiff(t1, t2) for t1, t2 in zip(cleaned_tags, single_annots.case_tags)
    ]
    isdirty = [len(t) > 0 for t in single_info['new_tags']]
    dirty_info = single_info.compress(isdirty)
    print('new_tags' + ut.repr4(ut.dict_hist(ut.flatten(single_info['new_tags']))))
    print('unioning new_tags into %r annots' % (len(dirty_info),))
    if not DRY:
        ibs.append_annot_case_tags(dirty_info['aid'], dirty_info['new_tags'])

    # Setup viewpoint
    mapping = [
        ('view-left', 'left'),
        ('view-right', 'right'),
        ('view-back', 'back'),
    ]
    single_info['viewpoint_code'] = [None] * len(single_info)
    for tag, yaw_text in mapping:
        tag_flags = ut.filterflags_general_tags(single_info['tags'], has_any=[tag])
        # setup yaw info
        for idx in ut.where(tag_flags):
            single_info['viewpoint_code'][idx] = yaw_text
    # Fix Viewpoint
    key1 = 'viewpoint_code'
    prop2 = 'viewpoint_code'
    repl2 = ('____', None)
    is_set = False
    check_annot_disagree(single_info, single_annots, key1, prop2, repl2, is_set, DRY=DRY)

    # Fix Names
    key1 = 'nameid'
    prop2 = 'names'
    repl2 = ('____', None)
    is_set = False
    check_annot_disagree(single_info, single_annots, key1, prop2, repl2, is_set, DRY=DRY)

    # Fix Species
    bad_flags = [s == '____' for s in single_annots.species]
    _annots = single_annots.compress(bad_flags)
    print('%d/%d annots need fixed species' % (sum(bad_flags), len(single_annots)))
    if not DRY:
        _annots.species = [species] * len(_annots)

    # Move injured/healthy/untagged to appropriate sets
    injur_tags = get_injured_tags(single_annots.case_tags, include_healthy=True)
    untagged = np.array(ut.lmap(len, injur_tags)) == 0
    untagged_annots = single_annots.compress(untagged)
    untagged_info = single_info.compress(untagged)
    print('%d/%d annots have no tags' % (len(untagged_annots), len(single_annots)))
    print(
        'Tags from WB imgnames:'
        + ut.repr3(ut.dict_hist(ut.flatten(untagged_info['tags'])))
    )
    untagged_images = ibs.images(untagged_annots.gids)
    # Add healthy tag to anything without an injured tag
    if not DRY:
        print('Adding healthy tag to sharked not taged as injured')
        ibs.append_annot_case_tags(
            untagged_annots.aids, ['healthy'] * len(untagged_annots.aids)
        )

    if not DRY:
        untagged_images.append_to_imageset('Untagged')

    categories = get_injur_categories(single_annots.case_tags)
    healthy_flags = ut.filterflags_general_tags(categories, any_startswith=('injur-'))
    injured_flags = ut.filterflags_general_tags(
        single_annots.case_tags, has_any=['healthy']
    )
    num_have = sum(ut.xor_lists(healthy_flags, injured_flags))
    num_miss = len(single_annots) - num_have
    print('missing %d annots' % (num_miss,))

    injured_annots = single_annots.compress(healthy_flags)
    injured_images = ibs.images(ut.unique(injured_annots.gids))
    healthy_annots = single_annots.compress(injured_flags)
    healthy_images = ibs.images(ut.unique(healthy_annots.gids))
    if not DRY:
        #
        injured_images.remove_from_imageset('Probably Healthy')
        healthy_images.remove_from_imageset('Probably Injured')
        #
        injured_images.append_to_imageset('Probably Injured')
        healthy_images.append_to_imageset('Probably Healthy')


def check_annot_disagree(
    single_info, single_annots, key1, prop2, repl2, is_set, key2=None, DRY=True
):
    info_prop = single_info[key1]
    if key2 is None:
        key2 = 'annot_' + prop2
        annot_prop = getattr(single_annots, prop2)
        annot_prop = [repl2[1] if p == repl2[0] else p for p in annot_prop]
        single_info[key2] = annot_prop
    else:
        annot_prop = single_info[key2]
    out_of_sync = [x != y for x, y in zip(info_prop, annot_prop)]
    if not is_set:

        def isnull(z):
            return z is None

    else:

        def isnull(z):
            return len(z) == 0

    ia_empty = [isnull(y) and not isnull(x) for x, y in zip(info_prop, annot_prop)]
    wb_empty = [isnull(x) and not isnull(y) for x, y in zip(info_prop, annot_prop)]
    disagree = [
        x != y and not isnull(x) and not isnull(y) for x, y in zip(info_prop, annot_prop)
    ]

    if is_set:
        # Like empty, but ia is a pure subset of wb
        ia_is_subset = [
            d and ut.issubset(y, x) for x, y, d in zip(info_prop, annot_prop, disagree)
        ]
        wb_is_subset = [
            d and ut.issubset(x, y) for x, y, d in zip(info_prop, annot_prop, disagree)
        ]
        # There may not be a subset, but there is overlap?
        some_isect = [
            d and len(ut.isect(y, x)) > 0
            for x, y, d in zip(info_prop, annot_prop, disagree)
        ]
        some_isect = ut.and_lists(some_isect, ut.not_list(ia_is_subset))
        some_isect = ut.and_lists(some_isect, ut.not_list(wb_is_subset))
        # Absolutely no overlap
        total_disagree = [
            d and len(ut.isect(y, x)) == 0
            for x, y, d in zip(info_prop, annot_prop, disagree)
        ]

    print('\n--- RECTIFY prop=%r --- ' % (key1,))

    print(
        'Prop=%r has %r/%r out of sync items' % (key1, sum(out_of_sync), len(out_of_sync))
    )
    print('WB has populated info for %r/%r %r' % (sum(ia_empty), len(ia_empty), key1))
    print('IA has populated info for %r/%r %r' % (sum(wb_empty), len(wb_empty), key1))
    print(
        'IA and WB disagree on info for %r/%r %r' % (sum(disagree), len(disagree), key1)
    )
    if is_set:
        print(
            'IA is subset of WB info for %r/%r %r'
            % (sum(ia_is_subset), len(ia_is_subset), key1)
        )
        print(
            'WB is subset of IA info for %r/%r %r'
            % (sum(wb_is_subset), len(wb_is_subset), key1)
        )
        print(
            'WB and IA partial overlap info for %r/%r %r'
            % (sum(some_isect), len(some_isect), key1)
        )
        print(
            'IA and WB total disagree on info for %r/%r %r'
            % (sum(total_disagree), len(total_disagree), key1)
        )

    sub_info = single_info.take_column('gid', 'aid', key1, key2)
    sub_info._meta['ignore'] = []

    print('\n--- DISAGREE DETAILS ---')

    print('IA POPULATED (updates on IA side?)')
    # Do nothing about these
    sub_info.compress(wb_empty).print()
    print('WB POPULATED: (can give)')
    # Pull info from wildbook
    sub_info.compress(ia_empty).print()
    if is_set:
        print('IA subset WB (can give)')
        sub_info.compress(ia_is_subset).print()
        print('WB subset IA (updates on IA side?)')
        sub_info.compress(wb_is_subset).print()
        print('SOME OVERLAP')
        # Have to manually fix
        sub_info.compress(some_isect).print()
        print('TOTAL DISAGREE')
        # Have to manually fix
        sub_info.compress(total_disagree).print()
    else:
        print('DISAGREE')
        # Have to manually fix
        sub_info.compress(disagree).print()

    # Get which annots need modification.
    if is_set:
        flags = ut.or_lists(ia_empty, ia_is_subset)
    else:
        # We can move populated info from wildbook into empty wbia info
        flags = ia_empty

    new_info = sub_info.compress(flags)
    old_annots = single_annots.compress(flags)

    if not is_set:
        # Ensure that there is no ambiguity
        isambiguous = [ut.isscalar(v) for v in new_info[key1]]
        notok = sum(ut.not_list(isambiguous))
        assert notok == 0
        print('There are %d ambiguous properties from wildbook' % (notok,))
        new_info = new_info.compress(isambiguous)
        old_annots = old_annots.compress(isambiguous)
        new_info = sub_info.compress(flags)
        old_annots = single_annots.compress(flags)

    new_prop = new_info[key1]
    if not is_set and len(ut.unique(new_prop)) < 20:
        print('new prop hist')
        print(ut.repr3(ut.dict_hist(new_prop)))
    elif is_set and len(ut.unique(ut.flatten(new_prop))) < 20:
        print('new prop hist')
        print(ut.repr3(ut.tag_hist(new_prop)))

    if not DRY:
        print('MODIFYING PROPERTEIS')
        if len(old_annots) > 0:
            if is_set:
                assert prop2 is None
                assert key1 == 'injur_tags', 'hack is invalid. got={}'.format(key1)
                old_annots.append_tags(new_prop)
            else:
                setattr(old_annots, prop2, new_prop)
    else:
        print('dryrun')


def get_injured_tags(tags_list, include_healthy=False, invert=False):
    """
    tags_list = single_info['tags']
    tags_list = single_annots.case_tags
    info_injur_tags = parse_injury_categories()
    annot_injur_tags = parse_injury_categories(single_annots.case_tags)
    """
    injur_patterns = [
        'injur-.*',
        'trunc',
        'nicks',
        'bite',
        'scar',
        '.*damage.*',
        '.*scar',
        '.*bite',
        'other_injury',
        'injured',
        'injur',
    ]
    if include_healthy:
        injur_patterns += ['healthy']
    flags_list = [
        [any([re.match(pat, t) for pat in injur_patterns]) for t in tags]
        for tags in tags_list
    ]
    if invert:
        flags_list = ut.lmap(ut.not_list, flags_list)
    only_injur_tags = ut.zipcompress(tags_list, flags_list)
    return only_injur_tags


def get_injur_categories(single_annots, verbose=False):
    # if verbose:
    #    print('Original tags')
    #    print(ut.repr3(ut.tag_hist(injur_tags)))

    if isinstance(single_annots, list):
        case_tags = single_annots
        aids = list(range(len(single_annots)))
    else:
        case_tags = single_annots.case_tags
        aids = single_annots.aids

    injur_tags = get_injured_tags(case_tags, include_healthy=True)

    cleaned_tags, alias_map, unmapped = ut.modify_tags(
        injur_tags,
        regex_map=[
            # Invalid patterns
            ('^.*' + re.escape('?') + '$', None),
            # Truncation
            ('injur-trunc', 'injur-trunc'),
            ('trunc', 'injur-trunc'),
            # Gill damage
            ('.*gilldamage.*', 'injur-gill'),
            # Other
            ('injur-unknown', 'injur-other'),
            ('injur-dead', 'injur-other'),
            ('other_injury', 'injur-other'),
            ('injur-damage', 'injur-other'),
            ('injured', 'injur-other'),
            ('^injur$', 'injur-other'),
            # Nicks
            ('nicks', 'injur-nicks'),
            ('injur-nicks-.*', 'injur-nicks'),
            ('.*bite', 'injur-bite'),
            ('.*scar', 'injur-scar'),
        ],
        direct_map=[
            ('injur-trunc', 'injur-trunc'),
            ('injur-scar', 'injur-scar'),
            ('injur-other', 'injur-other'),
            ('injur-nicks', 'injur-nicks'),
            ('injur-bite', 'injur-bite'),
            ('healthy', 'healthy'),
        ],
        return_unmapped=True,
        return_map=True,
        delete_unmapped=True,
    )
    assert len(unmapped) == 0, 'fixme %r' % (unmapped,)
    # Remove injur-other if other known injuries are present

    def fixinjur(aid, tags):
        tags = sorted(ut.unique(tags))
        injured = any([t.startswith('injur-') for t in tags])
        if injured:
            if 'healthy' in tags:
                print('shark aid=%r labeled as injured and healty %r!!!' % (aid, tags,))
        if len(tags) == 0:
            return tags
        tags = ut.setdiff(tags, ['injur-other'])
        if injured and len(tags) == 0:
            tags = ['injur-other']
        tags = ut.setdiff(tags, ['injur-gill'])
        if injured and len(tags) == 0:
            tags = ['injur-gill']
        # if len(tags) == 1:
        #    tags = ut.setdiff(tags, ['healthy'])
        return tags

    cleaned_tags = [fixinjur(aid, tags) for aid, tags in zip(aids, cleaned_tags)]
    if verbose:

        print(
            'mapping: ' + ut.repr3(ut.group_items(alias_map.keys(), alias_map.values()))
        )
        print('unmapped = %s' % (ut.repr3(unmapped),))

        given_tags = set(ut.flatten(injur_tags))
        alias_map_used = ut.odict()
        for val, key in alias_map.items():
            if val in given_tags:
                alias_map_used[val] = key
        print(
            'used_mapping: '
            + ut.repr3(ut.group_items(alias_map_used.keys(), alias_map_used.values()))
        )

        print('Cleaned tags')
        hist = ut.tag_hist(cleaned_tags)
        print(ut.repr3(hist))

        # Get tag co-occurrence
        print('Co-Occurrence Freq')
        co_occur = ut.tag_coocurrence(cleaned_tags)
        print(ut.repr3(co_occur))

        print('Co-Occurrence Percent')
        co_occur_percent = ut.odict(
            [
                (keys, [100 * val / hist[k] for k in keys])
                for keys, val in co_occur.items()
            ]
        )
        print(ut.repr3(co_occur_percent, precision=2, nl=1))

    # other_annots = single_annots.compress(ut.filterflags_general_tags(cleaned_tags, has_any=['injur-other']))
    # print('other_annots.case_tags = %s' % (ut.repr4(list(zip(other_annots.gids, other_annots.aids, other_annots.case_tags)), nl=1),))

    return cleaned_tags


def add_new_images(ibs, miss_info, species):
    import numpy as np

    isambiguous = miss_info.map_column('new_fpath', ut.isiterable)
    assert not any(isambiguous), 'Cannot add ambiguous filenames'

    # Add images to IA to get a gid
    gid_list = ibs.add_images(miss_info['new_fpath'])
    miss_info['gid'] = gid_list

    # Check to see if adding any images failed
    failed_flags = ut.flag_None_items(miss_info['gid'])
    print('# failed to add %s images' % (sum(failed_flags)),)

    passed_flags = ut.not_list(failed_flags)
    miss_info = miss_info.compress(passed_flags)

    ut.assert_all_not_None(miss_info['gid'])
    # ibs.get_image_uris_original(clist['gid'])

    assert (
        len(ut.find_duplicate_items(miss_info['gid'])) == 0
    ), 'duplicates should have already been sorted out'

    # Just choose one of the urls if any are ambiguous
    orig_urls = miss_info.map_column('img_url', lambda v: ut.ensure_iterable(v)[0])
    ibs.set_image_uris_original(miss_info['gid'], orig_urls, overwrite=True)

    print('Add new images to temporary imagesets')
    images_new = ibs.images(miss_info['gid'])
    new_imgsettext = 'New Images ' + ut.get_timestamp()
    images_new.append_to_imageset(new_imgsettext)
    injured_keywords = get_injured_tags(miss_info['tags'])
    hasinjur_kw = ut.lmap(bool, injured_keywords)
    images_new.compress(hasinjur_kw).append_to_imageset(new_imgsettext + ' Injur')
    images_new.compress(ut.not_list(hasinjur_kw)).append_to_imageset(
        new_imgsettext + ' Healthy'
    )

    verbose = True
    if verbose:
        other_keywords = get_injured_tags(miss_info['tags'], invert=True)
        print('Added %r new images' % (len(miss_info)))
        print('Of these, %r images had injured tags' % (sum(hasinjur_kw)))
        print('Of these, %r images had other tags' % (sum(ut.lmap(bool, other_keywords))))
        print(
            'Of these, %r images had no injured tags'
            % (len(miss_info) - sum(ut.lmap(bool, injured_keywords)))
        )
        # injured_keyhist = ut.dict_hist(ut.flatten(injured_keywords), ordered=True)
        # other_keyhist = ut.dict_hist(ut.flatten(other_keywords), ordered=True)
        # print('')
        # print('Injured Keyword histogram:\n' + ', '.join(
        #    ['*%s*: %s' % (k, v) for k, v in injured_keyhist.items()][::-1]))
        # print('')
        # print('Other Keyword histogram:\n' + ', '.join(
        #    ['*%s*: %s' % (k, v) for k, v in other_keyhist.items()][::-1]))

    is_empty_annots = np.array(images_new.num_annotations) == 0

    # Add anotations to images
    empty_new_info = miss_info.compress(is_empty_annots)
    empty_new_images = images_new.compress(is_empty_annots)

    # DETECT ANNOTATIONS ON NEW IMAGES
    if ibs.dbname == 'WS_ALL':
        # In the best case we have a detector
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

        images = ibs.images(empty_new_images.gids)
        images = images.compress([ext_ not in ['.gif'] for ext_ in images.exts])
        gid_list = images.gids

        # result is a tuple: (score, bbox_list, theta_list, conf_list, class_list)
        results_list = depc.get_property(
            'localizations', gid_list, None, config=config
        )  # NOQA
        print('Finished running localizations')

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
                # Take only a single annotation per bounding box.
                multi_gids.append(gid)
                idx = conf_list.argmax()
                res2 = (gid, bbox_list[idx : idx + 1], theta_list[idx : idx + 1])
                results_list2.append(res2)

        print('%d/%d have localizations' % (len(results_list2), len(results_list)))
        print('%d/%d are missing localizations' % (len(failed_gids), len(results_list)))
        print('%d/%d had multiple localizations' % (len(multi_gids), len(results_list)))

        # Add these to an imageset for fixing
        ibs.images(failed_gids).append_to_imageset('NoLocs' + new_imgsettext)
        ibs.images(multi_gids).append_to_imageset('MultiLocs' + new_imgsettext)

        # Reorder empty_info to be aligned with results
        localized_imgs = ibs.images(ut.take_column(results_list2, 0))
        empty_new_info_ = empty_new_info.loc_by_key('gid', localized_imgs.gids)
        assert all(
            [len(a) == 0 for a in localized_imgs.aids]
        ), 'no annots should be made yet'

        # Override old bboxes
        annot_gids = localized_imgs.gids
        annot_bboxes = np.array(ut.take_column(results_list2, 1))[:, 0, :]
        annot_thetas = np.array(ut.take_column(results_list2, 2))[:, 0]
        # Fix any ambiguities for name
        annot_names = empty_new_info_.map_column(
            'nameid', lambda v: ut.ensure_iterable(v)[0]
        )
        annot_names = ut.replace_nones(annot_names, ibs.const.UNKNOWN)
        # annot_names = empty_new_info_['nameid']
        annot_species = [species] * len(localized_imgs)
    else:
        # Make a single annotation for each image in the worst case
        annot_gids = empty_new_images.gids
        annot_bboxes = [(1, 1, w - 2, h - 2) for w, h in empty_new_images.sizes]
        annot_thetas = [0] * len(annot_gids)
        annot_names = empty_new_info.loc_by_key('gid', annot_gids)['nameid']
        annot_names = ut.replace_nones(annot_names, ibs.const.UNKNOWN)
        annot_species = [species] * len(annot_gids)

    aid_list = ibs.add_annots(
        annot_gids,
        bbox_list=annot_bboxes,
        theta_list=annot_thetas,
        name_list=annot_names,
        species_list=annot_species,
    )
    print('Finished adding new info')

    return aid_list


def _needs_redownload(fpath, seconds_thresh):
    if exists(fpath):
        file_info = ut.get_file_info(fpath)
        dt = ut.parse_timestamp(file_info['last_modified'], zone='UTC')
        delta = dt - ut.utcnow_tz()
        redownload = delta.total_seconds() > seconds_thresh
    else:
        redownload = True
    return redownload


def parse_whaleshark_org():
    """
    Read list of all images from wildbook

    Combines old and new

    >>> from wbia.scripts.getshark import *  # NOQA
    """
    from wbia.scripts import getshark

    parsed1 = getshark.parse_whaleshark_org_old()
    # Also parse using the keyword method
    parsed2 = getshark.parse_whaleshark_org_keywords()

    print('Parsed %d urls from XML jsp' % (len(parsed1),))
    print('Parsed %d urls from keywords' % (len(parsed2),))

    # Apply keywords to existing images
    # raise NotImplementedError('suffix is now unreliable for comparing encounters')

    # Use suffix as a key to create a merger mapping between indices
    print('Merging keyword and XML jsp results')
    suffix_to_idx1 = ut.make_index_lookup(parsed1['suffix'])
    suffix_to_idx2 = ut.make_index_lookup(parsed2['suffix'])
    idx1_to_idx2 = ut.dict_take(suffix_to_idx2, parsed1['suffix'], None)
    idx2_to_idx1 = ut.dict_take(suffix_to_idx1, parsed2['suffix'], None)

    # Find the items that are unique to each set
    unmatched_idx1 = ut.where(ut.not_list(idx1_to_idx2))
    unmatched_idx2 = ut.where(ut.not_list(idx2_to_idx1))
    print('There are %d unique entries in the XML results' % (len(unmatched_idx1),))
    print('There are %d unique entries in the jsp results' % (len(unmatched_idx2),))

    # nonmatching1 = parsed1.take(unmatched_idx1)
    # nonmatching2 = parsed2.take(unmatched_idx2)

    # Find the items that are common between both sets
    match_idx1 = ut.filter_Nones(idx2_to_idx1)
    match_idx2 = ut.filter_Nones(idx1_to_idx2)

    # matching1 = parsed1.take(match_idx1)
    # matching2 = parsed2.take(match_idx2)
    assert len(match_idx1) == len(match_idx2)
    print('There are %d items in common' % (len(match_idx1),))

    # Make columns agree between parsed1 and parsed2
    del parsed1['localid']
    parsed1['uuid'] = [None] * len(parsed1)
    parsed1['keywords'] = [[] for _ in range(len(parsed1))]
    ut.setdiff(parsed2.keys(), parsed1.keys())
    ut.setdiff(parsed1.keys(), parsed2.keys())

    parsed = parsed2 + parsed1
    parsed.cast_column('keywords', ut.oset)
    parsed.cast_column('new_fname', ut.ensure_iterable)
    parsed.cast_column('img_url', ut.ensure_iterable)
    parsed.cast_column('encounter', ut.ensure_iterable)
    parsed = parsed.merge_rows('suffix', merge_scalars=False)
    parsed.cast_column('keywords', list)
    parsed.cast_column('new_fname', lambda v: v[0])
    parsed.cast_column('img_url', lambda v: v[0])
    parsed.cast_column('encounter', lambda v: v[0])

    if True:
        parsed._meta['ignore'] = ['new_fname', 'img_url', 'suffix']
        parsed.print()

    # nonmatching1['nameid'] = [None] * len(nonmatching1)
    # nonmatching1['localid'] = [None] * len(nonmatching1)
    # Merge keywords from matching parts in parsed2 into parsed1
    # parsed1['keywords'] = [[] for _ in range(len(parsed1))]
    # for idx1, keys in zip(match_idx1, matching2['keywords']):
    #    parsed1['keywords'][idx1].extend(keys)

    # parsed = parsed2 + nonmatching1
    print('Parsed %d total urls' % (len(parsed),))
    return parsed


def parse_whaleshark_org_old():
    url = 'www.whaleshark.org/listImages.jsp'
    parsed1 = parse_wildbook_images(url)
    return parsed1


def parse_wildbook(images_url, keyword_url=None):
    """
    Read list of all images from wildbook

    Combines old and new

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.getshark import *  # NOQA
        >>> url = images_url = 'http://www.mantamatcher.org/listImages.jsp'

    Example:
        >>> # DISABLE_DOCTEST
        >>> images_url = 'http://www.whaleshark.org/listImages.jsp'
        >>> keyword_url = 'http://www.whaleshark.org/getKeywordImages.jsp'
    """
    from wbia.scripts import getshark

    parsed1 = getshark.parse_wildbook_images(images_url)
    # Also parse using the keyword method
    # parsed2 = getshark.parse_wildbook_keywords(keyword_url)

    print('Parsed %d urls from XML jsp' % (len(parsed1),))
    # if keyword_url:
    #     print('Parsed %d urls from keywords' % (len(parsed2),))

    # Apply keywords to existing images
    # raise NotImplementedError('suffix is now unreliable for comparing encounters')

    # Use suffix as a key to create a merger mapping between indices
    # print('Merging keyword and XML jsp results')
    # suffix_to_idx1 = ut.make_index_lookup(parsed1['suffix'])
    # suffix_to_idx2 = ut.make_index_lookup(parsed2['suffix'])
    # idx1_to_idx2 = ut.dict_take(suffix_to_idx2, parsed1['suffix'], None)
    # idx2_to_idx1 = ut.dict_take(suffix_to_idx1, parsed2['suffix'], None)

    # Find the items that are unique to each set
    # unmatched_idx1 = ut.where(ut.not_list(idx1_to_idx2))
    # unmatched_idx2 = ut.where(ut.not_list(idx2_to_idx1))
    # print('There are %d unique entries in the XML results' % (len(unmatched_idx1),))
    # print('There are %d unique entries in the jsp results' % (len(unmatched_idx2),))

    # nonmatching1 = parsed1.take(unmatched_idx1)
    # nonmatching2 = parsed2.take(unmatched_idx2)

    # Find the items that are common between both sets
    # match_idx1 = ut.filter_Nones(idx2_to_idx1)
    # match_idx2 = ut.filter_Nones(idx1_to_idx2)

    # assert len(match_idx1) == len(match_idx2)
    # print('There are %d items in common' % (len(match_idx1),))

    # Make columns agree between parsed1 and parsed2
    del parsed1['localid']
    parsed1['uuid'] = [None] * len(parsed1)
    parsed1['keywords'] = [[] for _ in range(len(parsed1))]

    parsed = parsed1
    parsed.cast_column('keywords', ut.oset)
    parsed.cast_column('new_fname', ut.ensure_iterable)
    parsed.cast_column('img_url', ut.ensure_iterable)
    parsed.cast_column('encounter', ut.ensure_iterable)
    parsed = parsed.merge_rows('suffix', merge_scalars=False)
    parsed.cast_column('keywords', list)
    parsed.cast_column('new_fname', lambda v: v[0])
    parsed.cast_column('img_url', lambda v: v[0])
    parsed.cast_column('encounter', lambda v: v[0])

    if True:
        parsed._meta['ignore'] = ['new_fname', 'img_url', 'suffix']
        parsed.print()

    # nonmatching1['nameid'] = [None] * len(nonmatching1)
    # nonmatching1['localid'] = [None] * len(nonmatching1)
    # Merge keywords from matching parts in parsed2 into parsed1
    # parsed1['keywords'] = [[] for _ in range(len(parsed1))]
    # for idx1, keys in zip(match_idx1, matching2['keywords']):
    #    parsed1['keywords'][idx1].extend(keys)

    # parsed = parsed2 + nonmatching1
    print('Parsed %d total urls' % (len(parsed),))
    return parsed


def parse_wildbook_images(url):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> url = 'www.whaleshark.org/listImages.jsp'
        >>> url = images_url = 'http://www.mantamatcher.org/listImages.jsp'
        >>> parse_wildbook_images(url)
    """
    from xml.dom.minidom import parseString
    from wbia.scripts import getshark

    number = None

    cache_dpath = ut.ensure_app_resource_dir('utool', 'sharkinfo')
    cache_fpath = join(cache_dpath, ut.hash_data(url) + '.xml')
    print('cache_fpath = {!r}'.format(cache_fpath))

    # redownload every 30 days or so
    if getshark._needs_redownload(cache_fpath, 60 * 60 * 24 * 30):
        XMLdata = ut.url_read_text(url)
        ut.writeto(cache_fpath, XMLdata)
    else:
        XMLdata = ut.readfrom(cache_fpath)

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

                new_fname = '%s-%i%s' % (nameid, localCount, ext)

                parsed_info['img_url'].append(img_url)
                parsed_info['nameid'].append(nameid)
                parsed_info['localid'].append(localCount)
                # might be different due to prefix
                parsed_info['new_fname'].append(new_fname)

                parsed_info['encounter'].append(encounter.getAttribute('number'))

                # print('Parsed %i / %i files.' % (len(parsed_info['orig_fname']), maxCount))
                if number is not None and len(parsed_info['orig_fname']) == number:
                    break

    parsed_ = ut.ColumnLists(parsed_info)
    print('Parsed %d urls from XML jsp' % (len(parsed_),))

    # Fix trivial (all non-keyword entries are the same) duplicates
    parsed_.cast_column('localid', lambda x: ut.oset(ut.ensure_iterable(x)))
    parsed_.cast_column('new_fname', lambda x: ut.oset(ut.ensure_iterable(x)))
    parsed1 = parsed_.merge_rows('img_url', merge_scalars=False)
    parsed1.cast_column('localid', lambda x: list(x)[0])
    parsed1.cast_column('new_fname', lambda x: list(x)[0])
    # Check and rectify for duplicate urls

    # Check and rectify for duplicate urls
    # unique_urls, idxs = parsed_.group_indicies('img_url')
    # toremove = []
    # for url, idxs in zip(unique_urls, idxs):
    #    if len(idxs) == 1:
    #        continue
    #    dupinfo = parsed_.take(idxs)
    #    del dupinfo[['localid', 'new_fname', 'img_url']]
    #    can_fix = True
    #    for key, vals in dupinfo.asdict().items():
    #        if not ut.allsame(vals):
    #            print(dupinfo.to_csv())
    #            print(('Duplicate items have different values'))
    #            # May need to fix a case when annoations happen in WB
    #            assert False, 'cant have this happen'
    #            can_fix = False
    #    if can_fix:
    #        toremove += idxs[1:]
    # print('Removing %d duplicate urls' % (len(toremove),))
    # flags = ut.not_list(ut.index_to_boolmask(toremove))
    # parsed1 = parsed_.compress(flags)

    prefix = commonprefix(parsed1['img_url'])
    parsed1['suffix'] = [url_[len(prefix) :] for url_ in parsed1['img_url']]
    return parsed1


def parse_whaleshark_org_keywords():
    verbose = True

    if verbose:
        print('[keywords] Parsing whaleshark.org keywords')

    # if False:
    #    key_url = 'http://www.whaleshark.org/getKeywordImages.jsp?indexName=nofilter&maxSize=2'
    #    #import requests
    #    #resp = requests.get(url)
    #    #resp.json()

    #    # key_url = 'http://www.whaleshark.org/getKeywordImages.jsp?indexName=nofilter'
    #    # json = cached_json_request(key_url)
    #    # ut.save_data('nofilterwbquery.pkl', json)

    #    #url = 'http://www.whaleshark.org/getKeywordImages.jsp?indexName=truncationleftpec&maxSize=2'
    #    #import requests
    #    #resp = requests.get(url)
    #    #resp.json()

    from wbia.scripts import getshark

    url = 'http://www.whaleshark.org/getKeywordImages.jsp'

    cache_dpath = ut.ensure_app_resource_dir('utool', 'sharkinfo3')

    def cached_json_request(key_url):
        import requests

        cache_fpath = join(cache_dpath, 'req_' + ut.hashstr27(key_url) + '.json')
        if getshark._needs_redownload(cache_fpath, 60 * 60 * 24 * 3000):
            print('Execute request %s' % (key_url,))
            resp = requests.get(key_url)
            print('Got Response')
            assert resp.status_code == 200
            dict_ = resp.json()
            ut.save_data(cache_fpath, dict_)
        else:
            dict_ = ut.load_data(cache_fpath)
        return dict_

    # Read all keyywords
    keywords = cached_json_request(url)['keywords']
    key_list = ut.take_column(keywords, 'indexName')
    if verbose:
        print('[keywords] Keyword indexName:')
        print(ut.indent('\n'.join(sorted(key_list)), '* '))

    # Request all images belonging to each keyword
    request_results = {}
    for key in ut.ProgIter(key_list + ['nofilter'], lbl='reading index', bs=False):
        key_url = url + '?indexName={indexName}'.format(indexName=key)
        request_results[key] = cached_json_request(key_url)

    keyed_images = {}
    for key, val in request_results.items():
        keyed_images[key] = val['images']

    # Flatten nested structure into ColumnList (note this will cause img_url duplicates)
    parsed_info2 = ut.ddict(list)
    for key, images in keyed_images.items():
        for imgdict in images:
            parsed_info2['img_url'].append(imgdict['url'])
            parsed_info2['encounter'].append(imgdict['correspondingEncounterNumber'])
            parsed_info2['nameid'].append(imgdict.get('individualID', None))
            parsed_info2['uuid'].append(imgdict['uuid'])
            parsed_info2['keywords'].append(imgdict['keywords'])
            # parsed_info2['keywords'].append([key])
    parsed2_ = ut.ColumnLists(parsed_info2)

    # Fix trivial (all non-keyword entries are the same) duplicates
    parsed2_.cast_column('keywords', ut.oset)
    parsed2 = parsed2_.merge_rows('img_url', merge_scalars=False)
    parsed2.cast_column('keywords', list)

    assert len(parsed2.get_multis('uuid')) == 0, 'uuids must be unique'

    if verbose:
        injured_keywords = getshark.get_injured_tags(parsed2['keywords'])
        other_keywords = getshark.get_injured_tags(parsed2['keywords'], invert=True)
        injured_keyhist = ut.dict_hist(ut.flatten(injured_keywords), ordered=True)
        other_keyhist = ut.dict_hist(ut.flatten(other_keywords), ordered=True)
        print('Scraped %r images with keywords' % (len(parsed2)))
        print(
            'Of these, %r images had injured tags'
            % (sum(ut.lmap(bool, injured_keywords)))
        )
        print('Of these, %r images had other tags' % (sum(ut.lmap(bool, other_keywords))))
        print(
            'Of these, %r images had no injured tags'
            % (len(parsed2) - sum(ut.lmap(bool, injured_keywords)))
        )

        print('')
        print(
            'Injured Keyword histogram:\n'
            + ', '.join(['*%s*: %s' % (k, v) for k, v in injured_keyhist.items()][::-1])
        )
        print('')
        print(
            'Other Keyword histogram:\n'
            + ', '.join(['*%s*: %s' % (k, v) for k, v in other_keyhist.items()][::-1])
        )

        # Get tag co-occurrence
        print('Injur Keywords Co-Occurrence Freq')
        co_occur = ut.tag_coocurrence(injured_keywords)
        print(ut.repr3(co_occur))
        print('Num co-occurrences: %r' % (sum(co_occur.values())))

        print('Injur Keywords Co-Occurrence Percent')
        co_occur_percent = ut.odict(
            [
                (keys, [100 * val / injured_keyhist[k] for k in keys])
                for keys, val in co_occur.items()
            ]
        )
        print(ut.repr3(co_occur_percent, precision=2, nl=1))

        _ = getshark.get_injur_categories(injured_keywords, verbose=True)  # NOQA

    prefix = commonprefix(parsed2['img_url'])
    parsed2['suffix'] = [url_[len(prefix) :] for url_ in parsed2['img_url']]
    # Hack off encounters so it aggrees with parsed1
    parsed2['suffix'] = [url_.lstrip('encounters/') for url_ in parsed2['suffix']]
    parsed2['new_fname'] = [suffix.replace('/', '--') for suffix in parsed2['suffix']]

    assert len(parsed2.get_multis('suffix')) == 0, 'hack invalidated something'
    return parsed2


def postprocess_filenames(parsed, download_dir):
    from os.path import commonprefix, basename  # NOQA

    # Create a new filename
    parsed['new_fpath'] = [join(download_dir, _fname) for _fname in parsed['new_fname']]
    # Remember the original filename
    prefix = commonprefix(parsed['img_url'])
    parsed['orig_fname'] = [url_[len(prefix) :] for url_ in parsed['img_url']]
    # Parse out the extension
    parsed['ext'] = [splitext(_fname)[-1] for _fname in parsed['new_fname']]
    return parsed


def postprocess_extfilter(parsed):
    # Filter based on image type (keep only jpgs)
    valid_exts = ['.jpg', '.jpeg', '.png']
    # , '.bmp', '.gif']
    ext_flags = [ext_.lower() in valid_exts for ext_ in parsed['ext']]

    invalid_exts = parsed.compress(ut.not_list(ext_flags))['ext']
    parsed = parsed.compress(ext_flags)
    num_removed = sum(ut.not_list(ext_flags))
    print('Invalid Extensions: ' + ut.repr3(ut.dict_hist(invalid_exts)))
    print('Valid Extensions: ' + ut.repr3(ut.dict_hist(parsed['ext'])))
    print('Removed %d images based on extensions' % (num_removed,))
    return parsed


def postprocess_tags_build(parsed):
    if False:
        parsed._meta['ignore'] = [
            'ext',
            'orig_fname',
            'new_fname',
            'img_url',
            'new_fpath',
            'encounter',
            'localid',
            'suffix',
            'nameid',
        ]
        parsed._meta['max_lines_start'] = 30
        parsed._meta['max_lines_end'] = 30
        parsed.print()

        parsed.compress(ut.and_lists(parsed['fname_tags'], parsed['keywords'])).print()

    # Filter to only images matching the appropriate tags
    from wbia.scripts import getshark

    parsed['fname_tags'] = getshark.parse_shark_fname_tags(parsed['orig_fname'])

    # Map keyword/fname tags to standard ia tags
    tags_list = ut.zipflat(parsed['fname_tags'], parsed['keywords'])
    cleaned_tags = ut.modify_tags(
        tags_list,
        direct_map=[('c429b13e4d232129014d251c74c60011', 'stranding'), ('', None)],
        regex_aug=[
            ('other_injury', 'injur-other'),
            ('truncation', 'injur-trunc'),
            ('nicks', 'injur-nicks'),
            ('scar', 'injur-scar'),
            ('bite', 'injur-bite'),
        ],
    )
    # cleaned_tags = ut.modify_tags(
    #    cleaned_tags,
    #    regex_aug=[
    #        ('injur-', 'injured'),
    #    ],
    # )
    parsed['tags'] = cleaned_tags
    return parsed


def postprocess_tags_filter(parsed):
    tag_flags = ut.filterflags_general_tags(
        parsed['tags'],
        # has_any=['view-left'],
        # none_match=['qual.*', 'view-top', 'part-.*', 'cropped'],
    )
    if all(tag_flags):
        print(
            'Tags histogram:'
            + ut.repr3(ut.dict_hist(ut.flatten(parsed['tags']), ordered=True))
        )
    else:
        print(
            'Tags before choosing:' + ut.repr3(ut.dict_hist(ut.flatten(parsed['tags'])))
        )
        parsed = parsed.compress(tag_flags)
        print('Tags after choosing:' + ut.repr3(ut.dict_hist(ut.flatten(parsed['tags']))))
    num_removed = sum(ut.not_list(tag_flags))
    print('Removed %d images based on tags' % (num_removed,))
    return parsed


def download_missing_images(parsed, num=None):
    exist_flags = ut.lmap(exists, parsed['new_fpath'])
    missing_flags = ut.not_list(exist_flags)
    print('nExist = %r / %r' % (sum(exist_flags), len(exist_flags)))
    print('nMissing = %r / %r' % (sum(missing_flags), len(exist_flags)))
    if any(missing_flags):
        missing = parsed.compress(missing_flags)
        print('Downloading missing subset')
        _iter = list(zip(missing['img_url'], missing['new_fpath']))
        if num:
            print('Only downloading {}'.format(num))

        from concurrent import futures

        ex = futures.ProcessPoolExecutor(7)
        fs = [
            ex.submit(ut.download_url, *args, new=True, verbose=False) for args in _iter
        ]

        for f in ut.ProgIter(
            futures.as_completed(fs),
            length=len(_iter),
            label='downloading wildbook images',
        ):
            pass

        # import multiprocessing
        # pool = multiprocessing.Pool(7)
        # res = pool.map(ut.partial(ut.download_url, new=True, verbose=False), _iter)

        # gen = ut.util_parallel.generate2(ut.download_url, zip(_iter), new=True, verbose=False)
        # for _ in gen:
        #     pass

        # _prog = ut.ProgPartial(bs=True, freq=1)
        # count = 0
        # for img_url, new_fpath in _prog(_iter, lbl='downloading wildbook images'):
        #     #url = img_url
        #     #filename = new_fpath
        #     #break
        #     try:
        #         ut.download_url(img_url, new_fpath, verbose=False, new=True)
        #         count += 1
        #         if num is not None and count > num:
        #             break
        #     except (ZeroDivisionError, IOError):
        #         pass


def postprocess_corrupted(parsed_dl):
    # Remove corrupted or ill-formatted images
    import vtool as vt

    print('Checking for corrupted images')
    fpaths = parsed_dl['new_fpath']
    valid_flags = vt.filterflags_valid_images(fpaths, verbose=2)
    parsed_dl = parsed_dl.compress(valid_flags)
    return parsed_dl


def postprocess_uuids(parsed_dl):
    # Assign uuids based on image content.
    # Stride of 1 is what IA uses internally
    print('Assigning file based UUID')
    _prog = ut.ProgPartial(bs=True, freq=10, adjust=True)
    parsed_dl['uuid'] = [
        ut.get_file_uuid(fpath_, stride=1)
        for fpath_ in _prog(parsed_dl['new_fpath'], lbl='uuid check')
    ]
    return parsed_dl


def postprocess_rectify_duplicates(unmerged):
    """ Rectify duplicate uuid information """
    print('Checking for duplicate information')
    # Find rows that have unique uuids
    singles = unmerged.get_singles('uuid')
    print('There are %d unique images that appear once' % (len(singles)))

    # Find rows with duplicate uuid entries
    multis = unmerged.get_multis('uuid')
    print('There are %d images that appear more than once' % (len(multis)))
    # Map other attributes to ordered-sets to join them
    multi_keys = ut.setdiff(unmerged.keys(), ['uuid'])
    multis.cast_column(multi_keys, lambda v: ut.oset(ut.ensure_iterable(v)))
    # Combine rows with the same uuid. (other attributes are set unioned)
    merged = multis.merge_rows('uuid')
    # Cast sets into lists
    merged.cast_column(multi_keys, list)
    print('There are %d unique images that have duplicates' % (len(merged)))

    # Rectify the duplicate information in the multi columns.
    # Tags/Keywords are simply unioned, leave them as is
    takeall_keys = ['tags', 'keywords', 'fname_tags']
    # Names/Encounters/etc should only take one value.
    # Try to fine one, but take multiple if it is ambiguous
    takeone_keys = ut.setdiff(multi_keys, takeall_keys)
    for key in takeone_keys:
        merged.cast_column(key, lambda v: v if len(v) <= 1 else ut.filter_Nones(v))
        merged.cast_column(key, lambda v: v[0] if len(v) == 1 else v)

    # We need to at least rectify some of the ambiguous information.
    # (ie, like where are we going to store the new image?)
    # For this just take the first item in the ambiguous list
    mustfix_keys = ['new_fpath', 'new_fname']
    isambiguous = ut.or_lists(*merged.map_column(mustfix_keys, ut.isiterable))
    for key in mustfix_keys:
        merged.cast_column(key, lambda v: ut.ensure_iterable(v)[0])

    print('Checking for ambiguous columns')
    for key in takeone_keys:
        isambiguous = merged.map_column(key, ut.isiterable)
        num = sum(isambiguous)
        if num > 0:
            ut.colorprint('X: key=%s has %r ambiguities!' % (key, num), 'red')
            # merged.compress(isambiguous).print(keys=[key, 'suffix'])
        else:
            ut.colorprint('o: key=%s is unambiguous' % (key,), 'green')

    # Take the first item from the columns that should only have one value
    # merged.cast_column(takeone_keys, lambda v: v[0])
    # Deal with animals with multiple names
    # merged.cast_column('nameid', lambda v: v if len(v) <= 1 else ut.filter_Nones(v))
    # merged.cast_column('nameid', lambda v: v[0] if len(v) == 1 else v)

    # print info
    # del parsed_dl[['ext', 'localid', 'orig_fname', 'suffix', 'new_fname', 'keywords']]
    if False:
        merged._meta['ignore'] = [
            'img_url',
            'orig_fname',
            'suffix',
            'new_fpath',
            'new_fname',
            'uuid',
            'ext',
            'fname_tags',
            'keywords',
        ]
        merged._meta['max_lines_start'] = 30
        merged._meta['max_lines_end'] = 30
        merged.print()

    # Combine and return the rectifyied information
    info = singles + merged
    print('Merged duplicates into %d truely unique images' % (len(info)))
    return info


def parse_shark_fname_tags(orig_fname_list, dev=False):
    """
    Parses potential tags from the filename. If dev mode is on, then it prints
    out other potential tags you might add.

    >>> orig_fname_list = parsed['orig_fname']
    >>> dev = True
    >>> tags = parse_shark_fname_tags(orig_fname_list, dev=dev)

    """
    import re

    invalid_tag_patterns = [
        re.escape('-'),
        re.escape('(') + '?\\d*' + re.escape(')') + '?',
        '\\d+-\\d+-\\d+',
        '\\d+,',
        '\\d+',
        'vi*',
        'i*v',
        'i+',
        '\\d+th',
        '\\d+nd',
        '\\d+rd',
        'remant',
        'timnfe',
        't',
        'e',
        'sjl',
        'disc',
        'dec',
        'road',
        'easter',
        'western',
        'west',
        'tn',
        '\\d*ap',
        'whaleshark\\d*',
        'shark\\d*',
        'whale\\d*',
        'whalesharking',
        'sharking',
        'whalesharks',
        'whales',
        'picture',
        'australien',
        'australia',
        'nick',
        'tim\\d*',
        'imageset',
        'holiday',
        'visit',
        'tour',
        'trip',
        'pec',
        'sv',
        'a',
        'b',
        'c',
        's',
        'd',
        'h',
        'g' 'gender',
        'sex',
        'img',
        'image',
        'pic',
        'pics',
        'leith',
        'trips',
        'kings',
        'photo',
        'video',
        'media',
        'fix',
        'feeding',
        'nrd',
        'nd',
        'gen',
        'wa',
        'nmp',
        'bo',
        'kd',
        'ow',
        'ne',
        'dsc',
        'nwd',
        'mg',
        'w',
        'mai',
        'blue',
        'stumpy',
        'oea',
        'cbe',
        'edc',
        'knrt',
        'tiws2',
        'ando',
        'adv',
        'str',
        'adventure',
        'camera',
        'tag',
        'id',
        'ws1',
        'ws',
        'gulf',
        'wally',
        'walhai',
        'wags',
        'shark[0-9][a-z]',
        'shark',
        'sharks',
        'reef',
        '720x480',
        'nb',
        'nrdive',
        'tiws',
        'exmouth',
        'nrdive2',
        'ningaloo',
        'ti',
        'nwss',
        '1st',
        'exp',
        'wsnd',
        'cba',
        '3iwsd',
        'c1',
        'nwd2',
        's1l1',
        's1r1' 'encounter',
        'of',
        'and',
        'the',
        'on',
        'to',
        'with',
        'in',
        'up',
        'ws3',
        's2' 'tagged',
        'from',
        'dive',
        'untag',
        'tagtrace',
        'day',
        '\\d*april',
        '\\d*may',
        '\\d*july',
        '\\d*june',
        'apr\\d+' 'ningaloo',
        'ningblue\\d*',
        'kooling',
    ]

    couldbe_tags = [
        'remnant',
        'prop',
        'north' 'shot',
        'professional',
        'red',
        'original',
        'measure',
        'gender',
        'encounter',
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
        ['injur-other', 'injury'],
        ['notch'],
        ['small'],
        ['qual-resize', 'resize'],
        ['qual-stretched', 'stretched'],
        ['pregnant'],
        ['notpregnant'],
        ['closeup'],
        ['mature'],
        ['ventralid'],
    ]

    cam_tags = [
        ['cam-slr2', 'slr2'],
        ['cam-5m', '5m'],
        ['cam-7m', '7m'],
        ['cam-4m', '4m'],
        ['copy'],
    ]

    invalid_tag_patterns += [re.escape(c) for c in ut.flatten(cam_tags)]
    # valid_tag_level_set += invalid_tag_patterns

    def apply_enum_regex(pat_list):
        enum_endings = [
            '[a-g]',
            '\\d*',
            'i*',
        ]
        expanded_pats = ut.flatten(
            [[pat + end for end in enum_endings] for pat in pat_list]
        )
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
        'view-left': apply_regex_endings(
            apply_enum_regex(inverse_alias_map['view-left'])
        ),
        'view-right': apply_regex_endings(
            apply_enum_regex(inverse_alias_map['view-right'])
        ),
    }

    valid_tags = list(inverse_alias_map.keys())

    invalid_tag_patterns = apply_regex_endings(invalid_tag_patterns)

    def parse_all_fname_tags(fname):
        from os.path import basename

        base = basename(splitext(fname)[0])
        # base.replace('(', '')
        # base.replace(')', '')
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

    known_img_tag_list = [
        list(set(tags).intersection(set(valid_tags))) for tags in all_img_tag_list
    ]

    if dev:
        # Help figure out which tags are important
        _parsed_tags = ut.flatten(all_img_tag_list)

        taghist = ut.dict_hist(_parsed_tags)
        taghist = {key: val for key, val in taghist.items() if val > 1}

        unknown_taghist = sorted(
            [(val, key) for key, val in taghist.items() if key not in valid_tags]
        )[::-1]
        known_taghist = sorted(
            [(val, key) for key, val in taghist.items() if key in valid_tags]
        )[::-1]

        print('Unknown')
        print(ut.repr2(unknown_taghist[0:100][::-1], nl=1))

        print('Known')
        print(ut.repr2(known_taghist[0:100][::-1], nl=1))

        print(
            ut.repr2(
                ut.dict_hist(ut.flatten(known_img_tag_list)), key_order_metric='val', nl=1
            )
        )

    return known_img_tag_list


# def main():
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


# def usage():
#    print('Fetches a number of images from the ECOCEAN shark database.')
#    print('Options:')
#    print('  -f <FILENAME> - Reads XML data from a file, rather than a URL.')
#    print('  -u <URL> - Reads XML data from the given URL.')
#    print('  -n <NUMBER> - Number of images to read; if omitted, reads all of them.')
#    print('  -h - Prints this help text.')


# if __name__ == '__main__':
#    main()
