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


    """
    CommandLine

        python -m ibeis.dev -e draw_rank_cdf --db PZ_MTEST --show -a ctrl:qsize=1 ctrl:qsize=3
        python -m ibeis.dev -e draw_rank_cdf -t candidacy_baseline --db PZ_MTEST -a ctrl --show
        python -m ibeis --tf -draw_rank_cdf -t candidacy_baseline -a ctrl --db PZ_MTEST --show
        python -m ibeis.dev -e draw_rank_cdf -t candidacy_invariance -a ctrl --db PZ_Master1 --show
        \
           --save invar_cumhist_{db}_a_{a}_t_{t}.png --dpath=~/code/ibeis/results  --adjust=.15 --dpi=256 --clipwhite --diskshow
        #ibeis -e rank_cdf --db lynx -a default:qsame_imageset=True,been_adjusted=True,excluderef=True -t default:K=1 --show

        python -m ibeis --tf draw_rank_cdf --db PZ_Master1 --show -t best:sv_on=[True,False] -a timectrlhard ---acfginfo --veryverbtd
        python -m ibeis --tf draw_rank_cdf --db PZ_Master1 --show -t best:refine_method=[homog,affine,cv2-homog,cv2-lmeds-homog] -a timectrlhard ---acfginfo --veryverbtd
        python -m ibeis --tf draw_rank_cdf --db PZ_Master1 --show -t best:refine_method=[homog,cv2-homog,cv2-lmeds-homog] -a timectrlhard ---acfginfo --veryverbtd

        python -m ibeis --tf draw_rank_cdf --db PZ_Master1 --show -t best -a timectrlhard:dsize=300 ---acfginfo --veryverbtd
        python -m ibeis --tf draw_match_cases --db PZ_Master1 -t best -a timectrlhard:dsize=300 ---acfginfo --veryverbtd --filt :orderby=gfscore,reverse=1,min_gtrank=1 --show
        python -m ibeis --tf draw_rank_cdf --db PZ_Master1 --show -t best -a timectrlhard:dsize=300 ---acfginfo --veryverbtd

        python -m ibeis.dev -e draw_rank_cdf --db PZ_Master1 --show -a ctrl -t default:lnbnn_on=True default:lnbnn_on=False,normonly_on=True default:lnbnn_on=False,bar_l2_on=True

        python -m ibeis.dev -e draw_rank_cdf --db PZ_MTEST --show -a ctrl -t default:lnbnn_on=True default:lnbnn_on=False,normonly_on=True default:lnbnn_on=False,bar_l2_on=True

        ibeis --tf draw_rank_cdf --db GZ_ALL -a ctrl -t default:K=1,resize_dim=[width,area],dim_size=[450,550] --show
        ibeis --tf autogen_ipynb --db GZ_ALL --ipynb -a ctrl:size=100 -t default:K=1,resize_dim=[width,area],dim_size=[450,550] --noexample

        ibeis --tf draw_rank_cdf --db GZ_ALL -a ctrl \
            -t default:K=1,resize_dim=[width],dim_size=[600,700,750] \
             default:K=1,resize_dim=[area],dim_size=[450,550,600,650] \
            --show

        ibeis --tf draw_rank_cdf --db GZ_ALL -a ctrl \
            -t default:K=1,resize_dim=[width],dim_size=[700,750] \
            --show
    """
