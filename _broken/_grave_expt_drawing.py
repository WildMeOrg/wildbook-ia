# BLIND CASES - draws results without labels to see if we can
# determine what happened using doubleblind methods
if DRAW_BLIND:
    pt.clf()
    best_gt_aid = cm.get_top_groundtruth_aid(ibs=ibs)
    cm.show_name_matches(
        ibs, best_gt_aid, show_matches=False,
        show_name_score=False, show_name_rank=False,
        show_annot_score=False, fnum=fnum, qreq_=qreq_,
        **show_kwargs)
    blind_figtitle = 'BLIND ' + query_lbl
    pt.set_figtitle(blind_figtitle)
    blind_fpath = join(individ_results_dpath, blind_figtitle) + '.png'
    pt.gcf().savefig(blind_fpath)
    #blind_fpath = pt.custom_figure.save_figure(fpath=blind_fpath, **dumpkw)
    cpq.append_copy_task(blind_fpath, blind_results_figdir)
    if metadata is not None:
        metadata.set_global_data(cfgstr, cm.qaid, 'blind_fpath', blind_fpath)

    DRAW_ANALYSIS = True
    DRAW_BLIND = False and not SHOW
