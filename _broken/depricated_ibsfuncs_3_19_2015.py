

@__injectable
def prune_exemplars(ibs):
    r"""
    Prunes exemplars from names with too many exemplars.

    Args:
        ibs (IBEISController):
    """
    nid_list = ibs.get_valid_nids()
    aids_list = ibs.get_name_exemplar_aids(nid_list)
    max_exemplars = ibs.cfg.other_cfg.max_exemplars
    problem_aids = [np.array(aids) for aids in aids_list if len(aids) > max_exemplars]
    problem_bboxes = unflat_map(ibs.get_annot_bboxes, problem_aids)
    #problem_gids   = unflat_map(ibs.get_annot_gids, problem_aids)
    #problem_sizes  = unflat_map(ibs.get_image_sizes, problem_gids)
    def bbox_area(bbox):
        return bbox[-2] * bbox[-1]
    def bboxes_area(bbox_list):
        return list(map(bbox_area, bbox_list))

    # Get area of annotations, area of parent images, and the ratio

    problem_annot_areas = list(map(np.array, list(map(bboxes_area, problem_bboxes))))

    #problem_img_areas = list(map(np.array, list(map(bboxes_area, problem_sizes))))

    #problem_ratios = [(annot_areas / img_areas) for annot_areas, img_areas in
    #                  zip(problem_annot_areas, problem_img_areas)]

    problem_sortx = [areas.argsort() for areas in problem_annot_areas]
    # Get aids with the smallest bounding boxes to unexemplar
    small_aids_list = [aids[sortx][:-max_exemplars] for aids, sortx in zip(problem_aids, problem_sortx)]
    small_aids = ut.flatten(small_aids_list)
    ibs.set_annot_exemplar_flags(small_aids, [False] * len(small_aids))


def export_testset_for_chuck(ibs, min_num_annots):
    """
    Exports a set with some number of annotations that has good demo examples.
    multiple annotations per name and large time variation within names.

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --dbdir /raid/work2/Turk/PZ_Master --min-num-annots 100
        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --dbdir /raid/work2/Turk/PZ_Master --min-num-annots 500

        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --dbdir /raid/work2/Turk/GZ_Master --min-num-annots 100
        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --dbdir
        /raid/work2/Turk/GZ_Master --min-num-annots 500_DOCTEST


        python -m ibeis.ibsfuncs --test-export_testset_for_chuck --db GIR_Tanya --min-num-annots 100

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> #dbdir = ut.get_argval(('--dbdir',), type_=str, default='testdb1')
        >>> min_num_annots = ut.get_argval(('--min-num-annots',), type_=int, default=500)
        >>> #ibs = ibeis.opendb('testdb1')
        >>> #ibs = ibeis.opendb(dbdir='/raid/work2/Turk/PZ_Master')
        >>> ibs = ibeis.opendb()  # dbdir=dbdir)
        >>> #ibs = ibeis.opendb(dbdir='/raid/work2/Turk/GZ_Master')
        >>> print(ibs.get_dbinfo_str())
        >>> #ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> result = export_testset_for_chuck(ibs, min_num_annots)
        >>> # verify results
        >>> print(result)

    min_num_annots = 500
    """
    import numpy as np

    min_num_annots_per_name = 3
    max_annot_per_image = 5
    #3

    #min_num_annots_per_name = 1
    #max_annot_per_image = 3000

    nid_list = ibs.get_valid_nids()
    aids_list = ibs.get_name_aids(nid_list)
    nAids_list = list(map(len, aids_list))
    nOther_aids_list = ibs.unflat_map(ibs.get_annot_num_contact_aids, aids_list)

    invalid_by_num_annots = [num < min_num_annots_per_name for num in nAids_list]
    invalid_by_num_others = [any([num > max_annot_per_image for num in nums])
                             for nums in nOther_aids_list]
    invalid_list = ut.or_lists(invalid_by_num_annots, invalid_by_num_others)

    valid_nids = ut.filterfalse_items(nid_list, invalid_list)

    def get_name_time_variation(ibs, nid_list):
        aids_list      = ibs.get_name_aids(nid_list)
        unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes, aids_list)
        unixtimes_arrs = list(map(np.array, unixtimes_list))
        fixtimes_list  = [arr[arr > 0] for arr in unixtimes_arrs]
        std_list       = [np.std(arr) if len(arr) > 1 else 0 for arr in fixtimes_list]
        return std_list

    std_list = get_name_time_variation(ibs, valid_nids)
    sorted_nids = ut.sortedby(valid_nids, std_list, reverse=True)

    # Find which names to include
    num_annot_cumsum = np.cumsum(ibs.get_name_num_annotations(sorted_nids))
    pos_list = np.where(num_annot_cumsum >= min_num_annots)[0]
    assert len(pos_list) > 0

    nid_list_chosen = sorted_nids[:pos_list[0] + 1]
    print('using names:')
    print(ibs.get_name_texts(nid_list_chosen))
    aids_list_chosen = ibs.get_name_aids(nid_list_chosen)
    aid_list_chosen = ut.flatten(aids_list_chosen)
    gid_list_chosen = ibs.get_annot_gids(aid_list_chosen)
    #ut.debug_duplicate_items(gid_list_chosen)

    # make sure not too many other annots are along for the ride
    other_aids = ibs.get_annot_contact_aids(aid_list_chosen)
    unexpected_aids = list(set(ut.flatten(other_aids)).difference(set(aid_list_chosen)))
    print('got %d unexpected_aids' % (len(unexpected_aids),))

    from ibeis.dbio import export_subset

    def new_nonconflicting_dbpath(ibs):
        dpath, dbname = split(ibs.get_dbdir())
        base_fmtstr = dbname + '_demo' + str(min_num_annots) + '_export%d'
        new_dbpath = ut.get_nonconflicting_path(base_fmtstr, dpath)
        return new_dbpath

    #ut.embed()

    dbpath = new_nonconflicting_dbpath(ibs)
    ibs_dst = ibeis.opendb(dbdir=dbpath, allow_newdir=True)
    ibs_src = ibs
    gid_list = gid_list_chosen
    export_subset.merge_databases(ibs_src, ibs_dst, gid_list=gid_list)

    DEBUG_NAME = False
    if DEBUG_NAME:
        ibs.get_name_num_annotations(sorted_nids[0:10])
        import plottool as pt
        ibeis.viz.viz_name.show_name(ibs, sorted_nids[0])
        pt.update()


def GreatZebraCount_batch_rename(ibs):
    """
    python dev.py --db PZ_MUGU_19 --cmd
    python dev.py --db PZ_MUGU_20 --cmd
    python dev.py --db PZ_MUGU_18 --cmd
    python dev.py --db GIRM_MUGU_20 --cmd
    """
    nid_list = ibs.get_valid_nids()
    name_list = ibs.get_name_texts(nid_list)
    new_name_list = [name.replace('IBEIS', 'MUGU') for name in name_list]
    ibs.set_name_texts(nid_list, new_name_list)
    print(ibs.get_name_texts(nid_list))
    ibs.cfg.other_cfg.location_for_names = 'MUGU'
    ibs.cfg.save()


def GreatZebraCount_mergedbs():
    """ docstr """

    import ibeis
    #ibeis.opendb('PZ_MUGU_18')
    #ibeis.opendb('GIRM_MUGU_20')

    ibs_pz_mugu_18 = ibeis.opendb('PZ_MUGU_18')
    ibs_pz_mugu_19 = ibeis.opendb('PZ_MUGU_19')
    ibs_pz_mugu_20 = ibeis.opendb('PZ_MUGU_20')

    ibs_pz_mugu_all  = ibeis.opendb('PZ_MUGU_ALL', allow_newdir=True)
    #ibs_mugu_all = ibeis.opendb('PZ_MUGU_ALL', allow_newdir=True, delete_ibsdir=True)

    def merge_rename(ibs, num):
        nid_list = ibs.get_valid_nids()
        name_list = ibs.get_name_texts(nid_list)
        assert all(['IBEIS' not in name for name in name_list])
        repl = 'MUGU_%d' % num
        new_name_list = [
            name if repl in name else name.replace('MUGU', repl)
            for name in name_list
        ]
        ibs.set_name_texts(nid_list, new_name_list)

    merge_rename(ibs_pz_mugu_18, 18)
    merge_rename(ibs_pz_mugu_19, 19)
    merge_rename(ibs_pz_mugu_20, 20)

    ibs_pz_mugu_18.print_name_table()
    ibs_pz_mugu_20.print_name_table()
    ibs_pz_mugu_19.print_name_table()
    ibs_pz_mugu_all.print_name_table()

    #ibeis.ibsfuncs.GreatZebraCount_batch_rename(ibs_pz_mugu_19)
    #ibeis.ibsfuncs.GreatZebraCount_batch_rename(ibs_pz_mugu_20)

    ibs_pz_mugu_18.fix_and_clean_database()
    ibs_pz_mugu_19.fix_and_clean_database()
    ibs_pz_mugu_20.fix_and_clean_database()
    ibs_pz_mugu_all.fix_and_clean_database()

    ibs_pz_mugu_18.print_dbinfo()
    ibs_pz_mugu_19.print_dbinfo()
    ibs_pz_mugu_20.print_dbinfo()
    ibs_pz_mugu_all.print_dbinfo()

    ibs_pz_mugu_19.print_infostr()
    ibs_pz_mugu_20.print_infostr()
    ibs_pz_mugu_all.print_infostr()

    # Merging works from {19}->{}
    ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_18, ibs_pz_mugu_all)
    ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_19, ibs_pz_mugu_all)
    # Remerging ignores due to the contributor from {19}->{19} This should be
    # changed. Images may be added and annots may be added also things might be
    # changed.
    #ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_19, ibs_mugu_all)
    ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_20, ibs_pz_mugu_all)

    len(ibs_pz_mugu_all.get_name_texts(ibs_pz_mugu_all.get_valid_nids()))
    ibs_pz_mugu_all.get_name_texts(ibs_pz_mugu_all.get_valid_nids())

    ####

    ibs_girm_mugu_20 = ibeis.opendb('GIRM_MUGU_20', allow_newdir=True)
    ibs_girm_mugu_20.fix_and_clean_database()

    # Master multi-species database
    ibs_pz_mugu_all = ibeis.opendb('PZ_MUGU_ALL')
    ibs_mugu_master = ibeis.opendb('MUGU_MASTER', allow_newdir=True)
    # Merge all PZ into Master
    ibeis.dbio.export_subset.merge_databases(ibs_pz_mugu_all, ibs_mugu_master)
    # Merge all GIRM into Master
    ibeis.dbio.export_subset.merge_databases(ibs_girm_mugu_20, ibs_mugu_master)
    ibs_mugu_master.print_dbinfo()
    ibs_mugu_master.print_dbinfo()
