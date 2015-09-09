def export_nnp_master3_subset(ibs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-export_nnp_master3_subset

    Example:
        >>> # SCRIPT
        >>> from ibeis.ibsfuncs import *  # NOQA
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
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=1
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=2
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=3
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=4
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=5
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=6
        python -m ibeis.ibsfuncs --test-make_temporally_distinct_blind_test --challenge-num=7

    Ignore:
        challenge_pdfs = ut.glob('.', 'pair_*_compressed.pdf', recursive=True)
        dname = 'localdata/all_challenges'
        ut.ensuredir(dname)
        ut.copy(challenge_pdfs, dname)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
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
