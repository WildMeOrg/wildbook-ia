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
#    # into the experiment_harness
#    locals_ = locals()
#    return locals_  # return in dict format for execstr_dict



