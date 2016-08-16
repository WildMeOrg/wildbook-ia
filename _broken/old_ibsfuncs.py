# -*- coding: utf-8 -*-
def export_nnp_master3_subset(ibs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-export_nnp_master3_subset

    Example:
        >>> # SCRIPT
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('NNP_Master3')
        >>> # execute function
        >>> result = export_nnp_master3_subset(ibs)
        >>> # verify results
        >>> print(result)
    """
    # Get the subset of the dataset marked as difficult
    annotmatch_rowid_list = ibs._get_all_annotmatch_rowids()
    ishard_list         = ibs.get_annotmatch_is_hard(annotmatch_rowid_list)
    isphotobomb_list    = ibs.get_annotmatch_is_photobomb(annotmatch_rowid_list)
    isscenerymatch_list = ibs.get_annotmatch_is_scenerymatch(annotmatch_rowid_list)
    isnondistinct_list  = ibs.get_annotmatch_is_nondistinct(annotmatch_rowid_list)
    hards        = np.array(ut.replace_nones(ishard_list, False))
    photobombs   = np.array(ut.replace_nones(isphotobomb_list, False))
    scenerys     = np.array(ut.replace_nones(isscenerymatch_list, False))
    nondistincts = np.array(ut.replace_nones(isnondistinct_list, False))
    flags = vt.and_lists(vt.or_lists(hards, nondistincts), ~photobombs, ~scenerys)
    annotmatch_rowid_list_ = ut.list_compress(annotmatch_rowid_list, flags)

    aid1_list = ibs.get_annotmatch_aid1(annotmatch_rowid_list_)
    aid2_list = ibs.get_annotmatch_aid2(annotmatch_rowid_list_)
    aid_list = sorted(list(set(aid1_list + aid2_list)))
    from ibeis import dbio
    gid_list = sorted(list(set(ibs.get_annot_gids(aid_list))))
    dbio.export_subset.export_images(ibs, gid_list, new_dbpath=join(ibs.get_workdir(), 'testdb3'))


def make_temporally_distinct_blind_test(ibs, challenge_num=None):

    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=1
        python -m ibeis.other.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=2
        python -m ibeis.other.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=3
        python -m ibeis.other.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=4
        python -m ibeis.other.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=5
        python -m ibeis.other.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=6
        python -m ibeis.other.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=7

    Ignore:
        challenge_pdfs = ut.glob('.', 'pair_*_compressed.pdf', recursive=True)
        dname = 'localdata/all_challenges'
        ut.ensuredir(dname)
        ut.copy(challenge_pdfs, dname)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('Elephants_drop1')
        >>> result = make_temporally_distinct_blind_test(ibs)
        >>> print(result)
    """
    # Parameters
    if challenge_num is None:
        challenge_num = ut.get_argval('--challenge-num', type_=int, default=1)
    num_repeats = 2
    num_singles = 4

    # Filter out ones from previous challenges
    all_aid_list = ibs.get_valid_aids()

    def get_challenge_aids(aid_list):
        # Get annots with timestamps
        aid_list_ = filter_aids_without_timestamps(ibs, aid_list)
        singletons, multitons = partition_annots_into_singleton_multiton(ibs, aid_list_)
        # process multitons
        hourdists_list = ibs.get_unflat_annots_hourdists_list(multitons)
        best_pairx_list = [vt.pdist_argsort(x)[-1] for x in hourdists_list]
        best_multitons = vt.ziptake(multitons, best_pairx_list)
        best_hourdists_list = ut.flatten(ibs.get_unflat_annots_hourdists_list(best_multitons))
        assert len(best_hourdists_list) == len(best_multitons)
        best_multitons_sortx = np.array(best_hourdists_list).argsort()[::-1]
        best_pairs = ut.list_take(best_multitons, best_multitons_sortx[0:num_repeats])
        best_multis = ut.flatten(best_pairs)

        # process singletons
        best_singles = ut.list_take(singletons, ut.random_indexes(len(singletons), num_singles, seed=0))
        best_singles = ut.flatten(best_singles)

        chosen_aids_ = best_multis + best_singles
        assert len(best_multis) == num_repeats * 2, 'len(best_multis)=%r' % (len(best_multis),)
        assert len(best_singles) == num_singles, 'len(best_singles)=%r' % (len(best_singles),)
        return chosen_aids_
        #return best_multis, best_singles

    aid_list = all_aid_list
    # Define invalid aids recusrively, so challenges are dijsoint
    invalid_aids = []
    for _ in range(1, challenge_num + 1):
        prev_chosen_aids_ = get_challenge_aids(aid_list)
        invalid_aids += prev_chosen_aids_
        aid_list = ut.setdiff_ordered(all_aid_list, invalid_aids)

    chosen_aids_ = get_challenge_aids(aid_list)
    chosen_nids_ = ibs.get_annot_nids(chosen_aids_)

    # Randomize order of chosen aids and nids
    shufflx = ut.random_indexes(len(chosen_aids_), seed=432)
    chosen_aids = ut.list_take(chosen_aids_, shufflx)
    chosen_nids = ut.list_take(chosen_nids_, shufflx)
    choosex_list = np.arange(len(chosen_aids))

    # ----------------------------------
    # SAMPLE A SET OF PAIRS TO PRESNET
    import itertools
    #nid_pair_list   = list(itertools.combinations(chosen_nids, 2))

    def index2d_take(list_, indexes_list):
        return [[list_[x] for x in xs] for xs in indexes_list]

    choosex_pair_list = list(itertools.combinations(choosex_list, 2))
    nid_pair_list = index2d_take(chosen_nids, choosex_pair_list)
    # Do not choose any correct pairs
    is_correct = [nid1 == nid2 for nid1, nid2 in nid_pair_list]
    p = 1. - np.array(is_correct)
    p /= p.sum()
    random_pair_sample = 15
    # Randomly remove nonmatch pairs
    pairx_list = np.arange(len(choosex_pair_list))
    num_remove = len(pairx_list) - random_pair_sample
    randstate = np.random.RandomState(seed=10)
    remove_xs = randstate.choice(pairx_list, num_remove, replace=False, p=p)
    keep_mask = np.logical_not(vt.index_to_boolmask(remove_xs, len(pairx_list)))
    sample_choosex_list = np.array(choosex_pair_list)[keep_mask]

    # One last shuffle of pairs
    #ut.random_indexes(len(chosen_aids), seed=10)
    randstate = np.random.RandomState(seed=10)
    randstate.shuffle(sample_choosex_list)

    # Print out info
    avuuids_list = ibs.get_annot_visual_uuids(chosen_aids)
    print('avuuids_list=\n' + ut.list_str(avuuids_list))
    chosen_avuuids_list = index2d_take(avuuids_list, sample_choosex_list)
    for count, (avuuid1, avuuid2)  in enumerate(chosen_avuuids_list, start=1):
        print('pair %3d: %s - vs - %s' % (count, avuuid1, avuuid2))
    # Print
    best_times = ibs.get_unflat_annots_hourdists_list(ibs.group_annots_by_name(chosen_aids_)[0])
    print('best_times = %r' % (best_times,))

    # -------------------------
    # WRITE CHOSEN AIDS TO DISK
    import ibeis.viz
    #fname = 'aidchallenge_' + ibs.get_dbname() + '_' + str(challenge_num)
    challengename = 'pair_challenge'  '_' + str(challenge_num) + '_dbname=' + ibs.get_dbname()
    output_dirname = ut.truepath(join('localdata', challengename))
    ut.ensuredir(output_dirname)
    #aid = chosen_aids[0]
    #ibeis.viz.viz_chip.show_many_chips(ibs, chosen_aids)

    def dump_challenge_fig(ibs, aid, output_dirname):
        import plottool as pt
        pt.clf()
        title_suffix = str(ibs.get_annot_visual_uuids(aid))
        fig, ax = ibeis.viz.show_chip(ibs, aid, annote=False, show_aidstr=False, show_name=False, show_num_gt=False, fnum=1, title_suffix=title_suffix)
        #fig.show()
        #pt.iup()
        fpath = ut.truepath(join(output_dirname, 'avuuid%s.png' % (title_suffix.replace('-', ''),)))
        pt.save_figure(fpath_strict=fpath, fig=fig, fnum=1, verbose=False)
        vt.clipwhite_ondisk(fpath, fpath)
        return fpath

    orig_fpath_list = [dump_challenge_fig(ibs, aid, output_dirname) for aid in chosen_aids]
    #fpath_pair_list = index2d_take(orig_fpath_list, choosex_pair_list)
    #fpath_pair_list_ = ut.list_compress(fpath_pair_list, keep_mask)
    fpath_pair_list_ = index2d_take(orig_fpath_list, sample_choosex_list)
    ut.vd(output_dirname)

    # ----- DUMP TO LATEX ---

    # Randomize pair ordering
    #randstate = np.random.RandomState(seed=10)
    ##ut.random_indexes(len(chosen_aids))
    #randstate.shuffle(fpath_pair_list_)

    #latex_blocks = [ut.get_latex_figure_str(fpath_pair, width_str='2.4in', nCols=2, dpath=output_dirname) for fpath_pair in fpath_pair_list_]
    latex_blocks = [ut.get_latex_figure_str(fpath_pair, width_str='.5\\textwidth', nCols=2, dpath=output_dirname, caption_str='pair %d' % (count,))
                    for count, fpath_pair in enumerate(fpath_pair_list_, start=1)]
    latex_text = '\n\n'.join(latex_blocks)
    #print(latex_text)
    title = challengename.replace('_', ' ')
    pdf_fpath = ut.compile_latex_text(latex_text, dpath=output_dirname, verbose=True, fname=challengename, silence=True, title=title)
    output_pdf_fpath = ut.compress_pdf(pdf_fpath)
    ut.startfile(output_pdf_fpath)
    """
    import ibeis
    ibs = ibeis.opendb('Elephants_drop1')

    Guesses:
        def verify_guesses(partial_pairs):
            aids_lists = ibs.unflat_map(ibs.get_annot_rowids_from_partial_vuuids, partial_pairs)
            aid_pairs = list(map(ut.flatten, aids_lists))
            aids1 = ut.get_list_column(aid_pairs, 0)
            aids2 = ut.get_list_column(aid_pairs, 1)
            truth = ibs.get_aidpair_truths(aids1, aids2)
            return truth

        # Chuck: 2, 6   --
        partial_pairs = [('fc4fcc', '47622'), ('981ef476', '73721a95')]
        print(verify_guesses(partial_pairs))
        # Mine: 1, 14   --
        partial_pairs = [('fc4f', 'f0d32'), ('73721', '47622')]
        print(verify_guesses(partial_pairs))
        # Hendrik: 3, X --
        partial_pairs = [('476225', '4c383f'), ('48c1', '981ef')]
        # Hendrik2: 3, 13 --
        partial_pairs = [('476225', '4c383f'), ('fc4fcc', '24a50bb')]
        print(verify_guesses(partial_pairs))
        # Tanya:
        partial_pairs = [('476225', 'fc4fcc48'), ('73721a', '981ef476')]

        partial_vuuid_strs = ut.flatten(partial_pairs)

        get_annots_from_partial_vuuids

    """
    #else:
    #    def format_challenge_pairs():
    #        # -------------------------
    #        # EMBED INTO A PDF
    #        #from utool.util_latex import *  # NOQA
    #        import utool as ut
    #        import vtool as vt
    #        #verbose = True
    #        dpath = '/home/joncrall/code/ibeis/aidchallenge'
    #        # Read images
    #        orig_fpath_list = ut.list_images(dpath, fullpath=True)
    #        # Clip white out
    #        clipped_fpath_list = [vt.clipwhite_ondisk(fpath) for fpath in orig_fpath_list]
    #        # move into temporary figure dir with hashed names
    #        fpath_list = []
    #        figdpath = join(dpath, 'figures')
    #        ut.ensuredir(figdpath)
    #        from os.path import splitext, basename, dirname
    #        for fpath in clipped_fpath_list:
    #            fname, ext = splitext(basename(fpath))
    #            fname_ = ut.hashstr(fname, alphabet=ut.ALPHABET_16) + ext
    #            fpath_ = join(figdpath, fname_)
    #            ut.move(fpath, fpath_)
    #            fpath_list.append(fpath_)

    #            figure_str = ut.get_latex_figure_str(fpath_list, width_str='2.4in', nCols=2, dpath=dirname(figdpath))
    #            input_text = figure_str
    #            pdf_fpath = ut.compile_latex_text(input_text, dpath=dpath, verbose=False, quiet=True)  # NOQA

    #        #output_pdf_fpath = ut.compress_pdf(pdf_fpath)

    #        #fpath_list
    #        ## Weirdness
    #        #new_rel_fpath_list = [ut.relpath_unix(fpath_, dpath) for fpath_ in new_fpath_list]

    #    #for fpath_pair in fpath_pair_list:
    #    #    print(fpath_pair)

    #    # -------------------------
    #    # EMBED INTO A PDF
    #if False:
    #    # TODO: quadratic programmming optimization
    #    # Actually its a quadratically constrained quadratic integer program
    #    def get_assignment_mat(ibs, aid_list):
    #        #import scipy.spatial.distance as spdist
    #        #nid_list = ibs.get_annot_nids(aid_list)
    #        aid1_list, aid2_list = list(zip(*ut.iprod(aid_list, aid_list)))
    #        truths = ibs.get_aidpair_truths(aid1_list, aid2_list)
    #        assignment_mat = truths.reshape(len(aid_list), len(aid_list))
    #        assignment_mat[np.diag_indices(assignment_mat.shape[0])] = 0
    #        return assignment_mat
    #    import scipy.spatial.distance as spdist
    #    pairwise_times = ibs.get_unflat_annots_hourdists_list([aid_list])[0]
    #    ptime_utri = np.triu(spdist.squareform(pairwise_times))  # NOQA
    #    assignment_utri = np.triu(get_assignment_mat(ibs, aid_list))  # NOQA
    #    nassign_utri = np.triu((1 - assignment_utri) - np.eye(len(assignment_utri)))  # NOQA

    #    r"""
    #    Let d be number of annots
    #    Let x \in {0, 1}^d be an assignment vector

    #    let l = num matching pairs = 2
    #    let m = num nonmatching pairs = 2

    #    # maximize total time delta
    #    min 1/2 x^T (-ptime_utri) x
    #    s.t.
    #    # match constraint
    #     .5 * x.T @ A @ x - l <= 0
    #    -.5 * x.T @ A @ x + l <= 0
    #    # nonmatch constraint
    #     .5 * x.T @ \bar{A} @ x - l <= 0
    #    -.5 * x.T @ \bar{A} @ x + l <= 0
    #    """

    #maxhourdists = np.array(list(map(ut.safe_max, hourdists_list)))
    #top_multi_indexes = maxhourdists.argsort()[::-1][:num_repeats]
    #top_multitons = ut.list_take(multitons, top_multi_indexes)

    #
    #scipy.optimize.linprog


#@__injectable
#def vacuum_and_clean_databases(ibs):
#    # Add to duct tape? or DEPRICATE
#    #ibs.vdd()
#    print(ibs.db.get_table_names())
#    # Removes all lblannots and lblannot relations as we are not using them
#    if False:
#        print(ibs.db.get_table_csv(const.NAME_TABLE))
#        print(ibs.db.get_table_csv(const.ANNOTATION_TABLE))
#        print(ibs.db.get_table_csv(const.LBLTYPE_TABLE))
#        print(ibs.db.get_table_csv(const.LBLANNOT_TABLE))
#        print(ibs.db.get_table_csv(const.AL_RELATION_TABLE))
#    if False:
#        # We deleted these all at one point, but its not a good operation to
#        # repeat
#        # Get old table indexes
#        #lbltype_rowids = ibs.db.get_all_rowids(const.LBLTYPE_TABLE)
#        lblannot_rowids = ibs.db.get_all_rowids(const.LBLANNOT_TABLE)
#        alr_rowids = ibs.db.get_all_rowids(const.AL_RELATION_TABLE)
#        # delete those tables
#        #ibs.db.delete_rowids(const.LBLTYPE_TABLE, lbltype_rowids)
#        ibs.db.delete_rowids(const.LBLANNOT_TABLE, lblannot_rowids)
#        ibs.db.delete_rowids(const.AL_RELATION_TABLE, alr_rowids)
#    ibs.db.vacuum()


def find_location_disparate_splits(ibs):
    """
    CommandLine:
        python -m ibeis.other.ibsfuncs --test-find_location_disparate_splits

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('NNP_Master3')
        >>> # execute function
        >>> offending_nids = find_location_disparate_splits(ibs)
        >>> # verify results
        >>> print('offending_nids = %r' % (offending_nids,))

    """
    import scipy.spatial.distance as spdist
    import functools
    #aid_list_count = ibs.get_valid_aids()
    aid_list_count = ibs.filter_aids_count()
    nid_list, gps_track_list, aid_track_list = ibs.get_name_gps_tracks(aid_list=aid_list_count)

    # Filter to only multitons
    has_multiple_list = [len(gps_track) > 1 for gps_track in gps_track_list]
    gps_track_list_ = ut.list_compress(gps_track_list, has_multiple_list)
    aid_track_list_ = ut.list_compress(aid_track_list, has_multiple_list)
    nid_list_ = ut.list_compress(nid_list, has_multiple_list)

    # Other properties
    unixtime_track_list_ = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, aid_track_list_)

    # Move into arrays
    gpsarr_track_list_ = list(map(np.array, gps_track_list_))
    unixtimearr_track_list_ = [np.array(unixtimes)[:, None] for unixtimes in unixtime_track_list_]

    def unixtime_hourdiff(x, y):
        return np.abs(np.subtract(x, y)) / (60 ** 2)

    haversin_pdist = functools.partial(spdist.pdist, metric=ut.haversine)
    unixtime_pdist = functools.partial(spdist.pdist, metric=unixtime_hourdiff)
    # Get distances
    gpsdist_vector_list = list(map(haversin_pdist, gpsarr_track_list_))
    hourdist_vector_list = list(map(unixtime_pdist, unixtimearr_track_list_))

    # Get the speed in kilometers per hour for each animal
    speed_vector_list = [gpsdist / hourdist for gpsdist, hourdist in
                         zip(gpsdist_vector_list, hourdist_vector_list)]

    #maxhourdist_list = np.array([hourdist_vector.max() for hourdist_vector in hourdist_vector_list])
    maxgpsdist_list  = np.array([gpsdist_vector.max() for gpsdist_vector in gpsdist_vector_list])
    maxspeed_list = np.array([speed_vector.max() for speed_vector in speed_vector_list])
    sortx  = maxspeed_list.argsort()
    sorted_maxspeed_list = maxspeed_list[sortx]
    #sorted_nid_list = np.array(ut.list_take(nid_list_, sortx))

    if False:
        import plottool as pt
        pt.plot(sorted_maxspeed_list)
        allgpsdist_list = np.array(ut.flatten(gpsdist_vector_list))
        alltimedist_list = np.array(ut.flatten(hourdist_vector_list))

        pt.figure(fnum1=1, doclf=True, docla=True)
        alltime_sortx = alltimedist_list.argsort()
        pt.plot(allgpsdist_list[alltime_sortx])
        pt.plot(alltimedist_list[alltime_sortx])
        pt.iup()

        pt.figure(fnum1=2, doclf=True, docla=True)
        allgps_sortx = allgpsdist_list.argsort()
        pt.plot(allgpsdist_list[allgps_sortx])
        pt.plot(alltimedist_list[allgps_sortx])
        pt.iup()

        #maxgps_sortx = maxgpsdist_list.argsort()
        #pt.plot(maxgpsdist_list[maxgps_sortx])
        pt.iup()

    maxgps_sortx = maxgpsdist_list.argsort()
    gpsdist_thresh = 15
    sorted_maxgps_list = maxgpsdist_list[maxgps_sortx]
    offending_sortx = maxgps_sortx.compress(sorted_maxgps_list > gpsdist_thresh)

    speed_thresh_kph = 6  # kilometers per hour
    offending_sortx = sortx.compress(sorted_maxspeed_list > speed_thresh_kph)
    #sorted_isoffending = sorted_maxspeed_list > speed_thresh_kph
    #offending_nids = sorted_nid_list.compress(sorted_isoffending)
    offending_nids = ut.list_take(nid_list_, offending_sortx)
    #offending_speeds = ut.list_take(maxspeed_list, offending_sortx)
    print('offending_nids = %r' % (offending_nids,))

    for index in offending_sortx:
        print('\n\n--- Offender index=%d ---' % (index,))
        # Inspect a specific index
        aids = aid_track_list_[index]
        nid = nid_list_[index]
        assert np.all(np.array(ibs.get_annot_name_rowids(aids)) == nid)

        aid1_list, aid2_list = zip(*list(ut.product(aids, aids)))
        annotmatch_rowid_list = ibs.get_annotmatch_rowid_from_superkey(aid1_list, aid2_list)
        annotmatch_truth_list = ibs.get_annotmatch_truth(annotmatch_rowid_list)
        annotmatch_truth_list = ut.replace_nones(annotmatch_truth_list, -1)
        truth_mat = np.array(annotmatch_truth_list).reshape((len(aids), len(aids)))

        contrib_rowids = ibs.get_image_contributor_rowid(ibs.get_annot_gids(aids))
        contrib_tags = ibs.get_contributor_tag(contrib_rowids)

        print('nid = %r' % (nid,))
        print('maxspeed = %.2f km/h' % (maxspeed_list[index],))
        print('aids = %r' % (aids,))
        print('gpss = %s' % (ut.list_str(gps_track_list_[index]),))
        print('contribs = %s' % (ut.list_str(contrib_tags),))

        print('speedist_mat = \n' + ut.numpy_str(spdist.squareform(speed_vector_list[index]), precision=2))
        truth_mat_str = ut.numpy_str(truth_mat, precision=2)
        truth_mat_str = truth_mat_str.replace('-1' , ' _')

        print('truth_mat = \n' + truth_mat_str)
        print('gpsdist_mat  = \n' + ut.numpy_str(spdist.squareform(gpsdist_vector_list[index]), precision=2))
        print('hourdist_mat = \n' + ut.numpy_str(spdist.squareform(hourdist_vector_list[index]), precision=2))

    return offending_nids

    #gpsdist_matrix_list = list(map(spdist.squareform, gpsdist_vector_list))

@__injectable
def find_offending_contributors(ibs):
    lat_min, lon_min = (-1.340726, 36.792234)
    lat_max, lon_max = (-1.341633, 36.793340)
    gid_list = ibs.get_valid_gids()
    gps_list = ibs.get_image_gps(gid_list)

    gid_list_filtered = [
        gid
        for gid, (lat, lon) in zip(gid_list, gps_list)
        if lat_min >= lat and lat >= lat_max  and lon_min <= lon and lon <= lon_max
    ]
    contrib_list_filtered = ibs.get_image_contributor_tag(gid_list_filtered)

    contribs = {}
    for gid, contrib in zip(gid_list_filtered, contrib_list_filtered):
        if contrib not in contribs:
            contribs[contrib] = []
        contribs[contrib].append(gid)

    lengths_list = list(zip(contribs.keys(), [ len(contribs[k]) for k in contribs.keys() ]))
    print(lengths_list)


def get_fastest_names(ibs, nid_list=None):
    r"""
    CommandLine:
        python -m ibeis.other.ibsfuncs --test-get_fastest_names

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> ibs = testdata_ibs('NNP_Master3')
        >>> nid_list = None
        >>> # execute function
        >>> nid_list_, maxspeed_list_ = get_fastest_names(ibs, nid_list)
        >>> # verify results
        >>> result = str(list(zip(nid_list_, maxspeed_list_))[0:10])
        >>> print(result)
    """
    if nid_list is None:
        nid_list = ibs._get_all_known_nids()
    maxspeed_list = ibs.get_name_max_speed(nid_list)
    # filter nans
    notnan_flags = ~np.isnan(maxspeed_list)
    maxspeed_list__ = maxspeed_list.compress(notnan_flags)
    nid_list__      = np.array(nid_list).compress(notnan_flags)
    # sort by speed
    sortx = maxspeed_list__.argsort()[::-1]
    nid_list_      = nid_list__.take(sortx)
    maxspeed_list_ = maxspeed_list__.take(sortx)
    return nid_list_, maxspeed_list_


def review_tagged_splits():
    """

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-review_tagged_splits --show
        python -m ibeis.annotmatch_funcs --exec-review_tagged_splits --show --db

    Example:
        >>> from ibeis.gui.guiback import *  # NOQA
        >>> import numpy as np
        >>> #back = testdata_guiback(defaultdb='PZ_Master1', activate=False)
        >>> #ibs = back.ibs
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> # Find aids that still need splits
        >>> aid_pair_list = ibs.filter_aidpairs_by_tags(has_any='SplitCase')
        >>> truth_list = ibs.get_aidpair_truths(*zip(*aid_pair_list))
        >>> _aid_list = ut.compress(aid_pair_list, truth_list)
        >>> _nids_list = ibs.unflat_map(ibs.get_annot_name_rowids, _aid_list)
        >>> _nid_list = ut.get_list_column(_nids_list, 0)
        >>> import vtool as vt
        >>> split_nids, groupxs = vt.group_indices(np.array(_nid_list))
        >>> problem_aids_list = vt.apply_grouping(np.array(_aid_list), groupxs)
        >>> #
        >>> split_aids_list = ibs.get_name_aids(split_nids)
        >>> assert len(split_aids_list) > 0, 'SPLIT cases are finished'
        >>> problem_aids = problem_aids_list[0]
        >>> aid_list = split_aids_list[0]
        >>> #
        >>> print('Run splits for tagd problem cases %r' % (problem_aids))
        >>> #back.run_annot_splits(aid_list)
        >>> print('Review splits for tagd problem cases %r' % (problem_aids))
        >>> from ibeis.viz import viz_graph
        >>> nid = split_nids[0]
        >>> selected_aids = np.unique(problem_aids.ravel()).tolist()
        >>> selected_aids = [] if ut.get_argflag('--noselect') else  selected_aids
        >>> print('selected_aids = %r' % (selected_aids,))
        >>> selected_aids = []
        >>> aids = ibs.get_name_aids(nid)
        >>> self = viz_graph.make_name_graph_interaction(ibs, aids=aids,
        >>>                                              with_all=False,
        >>>                                              selected_aids=selected_aids,
        >>>                                              with_images=True,
        >>>                                              prog='neato', rankdir='LR',
        >>>                                              augment_graph=False,
        >>>                                              ensure_edges=problem_aids.tolist())
        >>> ut.show_if_requested()

        rowids = ibs.get_annotmatch_rowid_from_superkey(problem_aids.T[0], problem_aids.T[1])
        ibs.get_annotmatch_prop('SplitCase', rowids)
        #ibs.set_annotmatch_prop('SplitCase', rowids, [False])
    """
    pass


def review_tagged_joins():
    """

    CommandLine:
        python -m ibeis.annotmatch_funcs --exec-review_tagged_joins --show --db PZ_Master1
        python -m ibeis.annotmatch_funcs --exec-review_tagged_joins --show --db testdb1

    Example:
        >>> from ibeis.gui.guiback import *  # NOQA
        >>> import numpy as np
        >>> import vtool as vt
        >>> #back = testdata_guiback(defaultdb='PZ_Master1', activate=False)
        >>> #ibs = back.ibs
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> # Find aids that still need Joins
        >>> aid_pair_list = ibs.filter_aidpairs_by_tags(has_any='JoinCase')
        >>> if ibs.get_dbname() == 'testdb1':
        >>>     aid_pair_list = [[1, 2]]
        >>> truth_list_ = ibs.get_aidpair_truths(*zip(*aid_pair_list))
        >>> truth_list = truth_list_ != 1
        >>> _aid_list = ut.compress(aid_pair_list, truth_list)
        >>> _nids_list = np.array(ibs.unflat_map(ibs.get_annot_name_rowids, _aid_list))
        >>> edge_ids = vt.get_undirected_edge_ids(_nids_list)
        >>> edge_ids = np.array(edge_ids)
        >>> unique_edgeids, groupxs = vt.group_indices(edge_ids)
        >>> problem_aids_list = vt.apply_grouping(np.array(_aid_list), groupxs)
        >>> problem_nids_list = vt.apply_grouping(np.array(_nids_list), groupxs)
        >>> join_nids = [np.unique(nids.ravel()) for nids in problem_nids_list]
        >>> join_aids_list = ibs.unflat_map(ibs.get_name_aids, join_nids)
        >>> assert len(join_aids_list) > 0, 'JOIN cases are finished'
        >>> problem_aid_pairs = problem_aids_list[0]
        >>> aid_list = join_aids_list[0]
        >>> #
        >>> print('Run JOINS for taged problem cases %r' % (problem_aid_pairs))
        >>> #back.run_annot_splits(aid_list)
        >>> print('Review splits for tagd problem cases %r' % (problem_aid_pairs))
        >>> from ibeis.viz import viz_graph
        >>> nids = join_nids[0]
        >>> selected_aids = np.unique(problem_aid_pairs.ravel()).tolist()
        >>> ut.flatten(ibs.get_name_aids(nids))
        >>> aids = ibs.sample_annots_general(ut.flatten(ibs.get_name_aids(nids)), sample_per_name=4, verbose=True)
        >>> import itertools
        >>> aids = ut.unique(aids + selected_aids)
        >>> self = viz_graph.make_name_graph_interaction(ibs, aids=aids, selected_aids=selected_aids, with_all=False, invis_edges=list(itertools.combinations(selected_aids, 2)))
        >>> #self = viz_graph.make_name_graph_interaction(ibs, nids, selected_aids=selected_aids)
        >>> ut.show_if_requested()

        rowids = ibs.get_annotmatch_rowid_from_superkey(problem_aids.T[0], problem_aids.T[1])
        ibs.get_annotmatch_prop('SplitCase', rowids)
        #ibs.set_annotmatch_prop('SplitCase', rowids, [False])
    """
    pass



