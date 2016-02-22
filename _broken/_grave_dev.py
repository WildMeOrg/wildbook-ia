#@devcmd('upsize', 'upscale')
#def up_dbsize_expt(ibs, qaid_list, daid_list=None):
#    """
#    Plots the scores/ranks of correct matches while varying the size of the
#    database.

#    Args:
#        ibs       (list) : IBEISController object
#        qaid_list (list) : list of annotation-ids to query

#    CommandLine:
#        python dev.py -t upsize --db PZ_MTEST --qaid 1:30:3 --cmd

#    Example:
#        >>> from ibeis.all_imports import *  # NOQA
#        >>> #ibs = ibeis.opendb('PZ_FlankHack')
#        >>> ibs = ibeis.opendb('PZ_MTEST')
#        >>> qaid_list = ibs.get_valid_aids()
#        >>> daid_list = None
#    """
#    print('updbsize_expt')
#    upsizekw = dict(
#        num_samp=utool.get_argval('--num-samples', int, 5),
#        clamp_gt=utool.get_argval('--clamp-gt', int, 1),
#        clamp_gf=utool.get_argval('--clamp-gf', int, 1),
#        seed=143039
#    )
#    upsizetup = ibs.get_upsize_data(qaid_list, daid_list, **upsizekw)
#    qaid_list, qaid_trues_list, qaid_false_samples_list, nTotal = upsizetup
#    # Create a progress marking function
#    progkw = dict(nTotal=nTotal, flushfreq=20, approx=False)
#    mark_, end_ = utool.log_progress('[upscale] progress: ',  **progkw)
#    count = 0
#    # Set up output containers and run test iterations
#    upscores_dict = utool.ddict(lambda: utool.ddict(list))
#    input_iter = zip(qaid_list, qaid_trues_list, qaid_false_samples_list)
#    # For each query annotation runs it as a query multiple times
#    # each time it increases the number of false annotation in the database
#    # so we can see how a score degrades as the number of false
#    # database annotations increases
#    for qaid, true_aids, false_aids_samples in input_iter:
#        #print('qaid = %r' % (qaid,))
#        #print('true_aids=%r' % (true_aids,))
#        # For each true match and false sample
#        for gt_aid, false_sample in utool.iprod(true_aids, false_aids_samples):
#            #print('  gt_aid=%r' % (gt_aid,))
#            #print('  false_sample=%r' % (false_sample,))
#            mark_(count)
#            count += 1
#            # Execute query
#            daids = false_sample + [gt_aid]
#            qres = ibs.query_chips([qaid], daids)[0]
#            # Elicit information
#            score = qres.get_gt_scores(gt_aids=[gt_aid])[0]
#            # Append result
#            upscores_dict[(qaid, gt_aid)]['dbsizes'].append(len(false_sample))
#            upscores_dict[(qaid, gt_aid)]['score'].append(score)
#    end_()

#    if not utool.get_argflag('--noshow'):
#        colors = pt.distinct_colors(len(upscores_dict))
#        pt.figure(fnum=1, doclf=True, docla=True)
#        for ix, ((qaid, gt_aid), upscores) in enumerate(upscores_dict.items()):
#            xdata = upscores['dbsizes']
#            ydata = upscores['score']
#            pt.plt.plot(xdata, ydata, 'o-', color=colors[ix])
#        figtitle = 'Effect of Database Size on Match Scores'
#        figtitle += '\n' + ibs.get_dbname()
#        figtitle += '\n' + ibs.cfg.query_cfg.get_cfgstr()
#        pt.set_figtitle(figtitle, font='large')
#        pt.set_xlabel('# Annotations in database')
#        pt.set_ylabel('Groundtruth Match Scores (annot-vs-annot)')
#        pt.dark_background()
#        dumpkw = {
#            'subdir'    : 'upsize',
#            'quality'   : False,
#            'overwrite' : True,
#            'verbose'   : 0
#        }
#        figdir = ibs.get_fig_dir()
#        ph.dump_figure(figdir, **dumpkw)

#    #---------
#    # Find highest
#    if False:
#        dbsample_index = 1
#        line_index = 0

#        highscore = 0
#        highpair = None
#        none_pairs = []
#        pair_list  = []
#        score_list = []
#        for pair, dict_ in six.iteritems(upscores_dict):
#            scores = dict_['score']
#            if any([s is None for s in scores]):
#                none_pairs.append(pair)
#            if dbsample_index >= len(scores):
#                continue
#            score = scores[dbsample_index]
#            if score is None:
#                continue
#            score_list.append(score)
#            pair_list.append(pair)

#        sorted_tups = sorted(list(zip(score_list, pair_list)))
#        print(sorted_tups[0])
#        print(sorted_tups[-1])

#        qaid, gt_aid = sorted_tups[line_index][1]
#        print('qaid = %r' % qaid)
#        print('gt_aid = %r' % gt_aid)
#        index = qaid_list.index(qaid)
#        print(index)
#        false_aids_samples = qaid_false_samples_list[index]
#        false_sample = false_aids_samples[dbsample_index]
#        print(false_sample)
#        daids = false_sample + [gt_aid]
#        qres = ibs.query_chips([qaid], daids)[0]
#        #for score in scores:
#        #    if score is None:
#        #        continue
#        #    if score > highscore:
#        #        highpair = pair
#        #        highscore = score
#        #print(scores)

#    # TODO: Should be separate function. Previous code should be intergrated
#    # into the harness
#    locals_ = locals()
#    return locals_  # return in dict format for execstr_dict




@register_ibs_method
def get_annot_rowid_sample(ibs, aid_list=None, per_name=1, min_gt=1,
                           method='random', seed=0, offset=0,
                           stagger_names=False, distinguish_unknowns=True,
                           grouped_aids=None):
    r"""
    DEPRICATE

    Gets a sampling of annotations

    Args:
        per_name (int): number of annotations per name
        min_ngt (int): filters any name with less than this number of annotations
        seed (int): random seed
        aid_list (list): base aid_list to start with. If None
        get_valid_aids(minqual='poor') is used stagger_names (bool): if True
        staggers the order of the returned sample

    Returns:
        list: sample_aids

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_annot_rowid_sample

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> per_name = 3
        >>> min_gt = 1
        >>> seed = 0
        >>> # execute function
        >>> sample_aid_list = ibs.get_annot_rowid_sample(None, per_name=per_name, min_gt=min_gt, seed=seed)
        >>> result = ut.hashstr_arr(sample_aid_list)
        arr((66)crj9l5jde@@hdmlp)
    """
    #qaids = ibs.get_easy_annot_rowids()
    if grouped_aids is None:
        if aid_list is None:
            aid_list = np.array(ibs.get_valid_aids(minqual='poor'))
        grouped_aids_, unique_nids = ibs.group_annots_by_name(
            aid_list, distinguish_unknowns=distinguish_unknowns)
        if min_gt is None:
            grouped_aids = grouped_aids_
        else:
            grouped_aids = list(filter(lambda x: len(x) >= min_gt, grouped_aids_))
    else:
        # grouped aids was precomputed
        pass
    if method == 'random2':
        # always returns per_name when available
        sample_aids_list = ut.sample_lists(grouped_aids, num=per_name, seed=seed)
    elif method == 'random':
        # Random that allows for offset.
        # may return less than per_name when available if offset > 0
        rng = np.random.RandomState(seed)
        for aids in grouped_aids:
            rng.shuffle(aids)
        sample_aids_list = ut.get_list_column_slice(grouped_aids, offset, offset + per_name)
    elif method == 'simple':
        sample_aids_list = ut.get_list_column_slice(grouped_aids, offset, offset + per_name)
    else:
        raise NotImplementedError('method = %r' % (method,))
    if stagger_names:
        from six.moves import zip_longest
        sample_aid_list = ut.filter_Nones(ut.iflatten(zip_longest(*sample_aids_list)))
    else:
        sample_aid_list = ut.flatten(sample_aids_list)

    return sample_aid_list


@register_ibs_method
def get_one_annot_per_name(ibs, col='rand'):
    r"""

    DEPRICATE

    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.ibsfuncs --test-get_one_annot_per_name --db PZ_Master0
        python -m ibeis.ibsfuncs --test-get_one_annot_per_name --db PZ_MTEST
        python -m ibeis.ibsfuncs --test-get_one_annot_per_name --dbdir /raid/work2/Turk/GIR_Master

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> result = get_one_annot_per_name(ibs)
        >>> # verify results
        >>> print(result)
    """
    #nid_list = ibs.get_valid_nids()
    #aids_list = ibs.get_name_aids(nid_list)
    #num_annots_list = list(map(len, aids_list))
    #aids_list = ut.sortedby(aids_list, num_annots_list, reverse=True)
    #aid_list = ut.get_list_column(aids_list, 0)
    # Keep only a certain number of annots for distinctiveness mapping
    #aid_list_ = ut.listclip(aid_list, max_annots)
    aid_list_ = ibs.get_valid_aids()
    aids_list, nid_list = ibs.group_annots_by_name(aid_list_, distinguish_unknowns=True)
    if col == 'rand':
        def random_choice(aids):
            size = min(len(aids), 1)
            return np.random.choice(aids, size, replace=False).tolist()
        aid_list = [random_choice(aids)[0] if len(aids) > 0 else [] for aids in aids_list]
    else:
        aid_list = ut.get_list_column(aids_list, 0)
    allow_unnamed = True
    if not allow_unnamed:
        raise NotImplementedError('fixme')
    if col == 'rand':
        import random
        random.shuffle(aid_list)
    return aid_list


@register_ibs_method
def get_annot_groundfalse_sample(ibs, aid_list, per_name=1, seed=False):
    """
    get_annot_groundfalse_sample

    FIXME
    DEPRICATE

    Args:
        ibs (IBEISController):
        aid_list (list):
        per_name (int): number of groundfalse per name
        seed (bool or int): if False no seed, otherwise seeds numpy randgen

    Returns:
        list: gf_aids_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.ibsfuncs import *  # NOQA
        >>> import ibeis  # NOQA
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()[::4]
        >>> per_name = 1
        >>> seed = 42
        >>> sample_trues_list = get_annot_groundfalse_sample(ibs, aid_list, per_name, seed)
        >>> #result = str(sample_trues_list)
        >>> #print(result)

    [[3, 5, 7, 8, 10, 12, 13], [3, 7, 8, 10, 12, 13], [3, 6, 7, 8, 10, 12, 13], [2, 6, 7, 8, 10, 12]]
    [[2, 6, 7, 8, 10, 12, 13], [2, 7, 8, 10, 12, 13], [2, 5, 7, 8, 10, 12, 13], [2, 6, 7, 8, 10, 12]]
    [[2, 5, 7, 8, 10, 12, 13], [3, 7, 8, 10, 12, 13], [2, 5, 7, 8, 10, 12, 13], [3, 5, 7, 8, 10, 12]]
    """
    if seed is not False:
        # Determanism
        np.random.seed(seed)
    # Get valid names
    valid_aids = ibs.get_valid_aids()
    valid_nids = ibs.get_annot_name_rowids(valid_aids)
    nid2_aids = ut.group_items(valid_aids, valid_nids)
    for nid in list(nid2_aids.keys()):
        if ibs.is_nid_unknown(nid):
            # Remove unknown
            del nid2_aids[nid]
            continue
        # Cast into numpy arrays
        aids =  np.array(nid2_aids[nid])
        if len(aids) == 0:
            # Remove empties
            print('[ibsfuncs] name with 0 aids. need to clean database')
            del nid2_aids[nid]
            continue
        nid2_aids[nid] = aids
        # Shuffle known annotations in each name
        #np.random.shuffle(aids)
    # Get not beloning to input names
    nid_list = ibs.get_annot_name_rowids(aid_list)
    def _sample(nid_):
        aids_iter = (aids for nid, aids in six.iteritems(nid2_aids) if nid != nid_)
        sample_gf_aids = np.hstack([np.random.choice(aids, per_name,
                                                     replace=False) for aids in
                                    aids_iter])
        return sample_gf_aids.tolist()
    gf_aids_list = [_sample(nid_) for nid_ in nid_list]
    return gf_aids_list





def hack(ibs):
    #ibs.get_imageset_text(imgsetid_list)
    #imgsetid = ibs.get_imageset_imgsetids_from_text("NNP GZC Car '1PURPLE'")

    def get_name_linked_imagesets_by_imgsetid(ibs, imgsetid):
        import utool as ut
        #gid_list = ibs.get_imageset_gids(imgsetid)
        aid_list_ = ibs.get_imageset_aids(imgsetid)
        aid_list = ut.filterfalse_items(aid_list_, ibs.is_aid_unknown(aid_list_))

        #all(ibs.db.check_rowid_exists(const.ANNOTATION_TABLE, aid_list))
        #aids_list2 = ibs.get_image_aids(gid_list)
        #assert ut.flatten(aids_list2) == aids_list1
        nid_list = list(set(ibs.get_annot_nids(aid_list, distinguish_unknowns=False)))
        # remove unknown annots
        name_imgsetids = ibs.get_name_imgsetids(nid_list)
        name_imagesettexts = ibs.get_imageset_text(name_imgsetids)
        return name_imagesettexts

    imgsetid_list = ibs.get_valid_imgsetids()
    linked_imagesettexts = [get_name_linked_imagesets_by_imgsetid(ibs, imgsetid) for imgsetid in imgsetid_list]
    imagesettext_list = ibs.get_imageset_text(imgsetid_list)
    print(ut.dict_str(dict(zip(imgsetid_list, linked_imagesettexts))))
    print(ut.align(ut.dict_str(dict(zip(imagesettext_list, linked_imagesettexts))), ':'))
    print(ut.align(ut.dict_str(dict(zip(imagesettext_list, imgsetid_list)), sorted_=True), ':'))

    #if False:
    #    imgsetids_with_bad_names = [6, 7, 16]
    #    bad_nids = ut.unique_ordered(ut.flatten(ibs.get_imageset_nids(imgsetids_with_bad_names)))

