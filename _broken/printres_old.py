    else:
        chunksize = 4
        # <FOR RCITER_CHUNK>
        #with ut.EmbedOnException():
        def is_skipped(count):
            return (count in skip_list) or (SKIP_TO and count < SKIP_TO)

        total = len(sel_cols) * len(sel_rows)
        rciter = list(itertools.product(sel_rows, sel_cols))
        for rciter_chunk in ut.ichunks(enumerate(rciter), chunksize):
            # First load a chunk of query results
            # <FOR RCITER>
            qreq_list = [cfgx2_qreq_[c] for count, (r, c) in rciter_chunk if not is_skipped(count)]
            qres_list = [load_qres(ibs, qaids[r], daids, cfgx2_qreq_[c])
                         for count, (r, c) in rciter_chunk if not is_skipped(count)]

            # Iterate over chunks a second time, but
            # with loaded query results
            for (count, rctup), qres, qreq_ in zip(rciter_chunk, qres_list, qreq_list):
                if (count in skip_list) or (SKIP_TO and count < SKIP_TO):
                    continue
                (r, c) = rctup
                fnum = c if SHOW else 1
                # Get row and column index
                query_lbl = cfgx2_lbl[c]
                print(ut.unindent('''
                __________________________________
                --- VIEW %d / %d --- (r=%r, c=%r)
                ----------------------------------
                ''')  % (count + 1, total, r, c))
                qres_cfg = qres.get_fname(ext='')
                subdir = qres_cfg
                # Draw Result
                dumpkw = {
                    'subdir'    : subdir,
                    'quality'   : QUALITY,
                    'overwrite' : True,
                    'verbose'   : 0,
                }
                show_kwargs = {
                    'N': 3,
                    'ori': True,
                    'ell_alpha': .9,
                }

                #if not SAVE_FIGURES:
                #    continue

                #if USE_FIGCACHE and ut.checkpath(join(figdir, subdir)):
                #    pass

                print('[harn] drawing analysis plot')

                # Show Figure
                # try to shorten query labels a bit
                query_lbl = query_lbl.replace(' ', '').replace('\'', '')
                #qres.show(ibs, 'analysis', figtitle=query_lbl, fnum=fnum, **show_kwargs)
                if SHOW:
                    qres.ishow_analysis(ibs, figtitle=query_lbl, fnum=fnum, annot_mode=1, qreq_=qreq_, **show_kwargs)
                    #qres.show_analysis(ibs, figtitle=query_lbl, fnum=fnum, annot_mode=1, qreq_=qreq_, **show_kwargs)
                else:
                    qres.show_analysis(ibs, figtitle=query_lbl, fnum=fnum, annot_mode=1, qreq_=qreq_, **show_kwargs)

                # Adjust subplots
                #df2.adjust_subplots_safe()

                if SHOW:
                    print('[DRAW_RESULT] df2.present()')
                    # Draw only once we finish drawing all configs (columns) for
                    # this row (query)
                    if c == len(sel_cols) - 1:
                        #execstr = df2.present()  # NOQA
                        ans = input('press to continue...')
                        if ans == 'cmd':
                            ut.embed()
                        #six.exec_(execstr, globals(), locals())
                        #exec(df2.present(), globals(), locals())
                    #print(execstr)
                # Saving will close the figure
                fpath_orig = ph.dump_figure(figdir, reset=not SHOW, **dumpkw)
                append_copy_task(fpath_orig)

                print('[harn] drawing extra plots')

                DUMP_QANNOT         = DUMP_EXTRA
                if DUMP_QANNOT:
                    _show_chip(qres.qaid, 'QUERY_', config2_=qreq_.qparams, **dumpkw)
                    _show_chip(qres.qaid, 'QUERY_CXT_', in_image=True, config2_=qreq_.extern_query_config2, **dumpkw)

                DUMP_QANNOT_DUMP_GT = DUMP_EXTRA
                if DUMP_QANNOT_DUMP_GT:
                    gtaids = ibs.get_annot_groundtruth(qres.qaid)
                    for aid in gtaids:
                        rank = qres.get_aid_ranks(aid)
                        _show_chip(aid, 'GT_CXT_', rank=rank, in_image=True, config2_=qreq_.extern_data_config2, **dumpkw)

                DUMP_TOP_CONTEXT    = DUMP_EXTRA
                if DUMP_TOP_CONTEXT:
                    topids = qres.get_top_aids(num=3)
                    for aid in topids:
                        rank = qres.get_aid_ranks(aid)
                        _show_chip(aid, 'TOP_CXT_', rank=rank, in_image=True, config2_=qreq_.extern_data_config2, **dumpkw)
            flush_copy_tasks()
        flush_copy_tasks()
        # </FOR RCITER>
