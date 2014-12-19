from __future__ import absolute_import, division, print_function
import utool as ut
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[user_dialogs]')


def wait_for_user_name_decision(ibs, qres, qreq_, autoname_msg, name,
                                      name_confidence, choicetup, incinfo=None):
    r""" hooks into to some method of getting user input for names

    TODO: really good interface

    Prompts the user for input

    Args:
        ibs (IBEISController):
        qres (QueryResult):  object of feature correspondences and scores
        autoname_func (function):
    """
    if qres is None:
        print('WARNING: qres is None')
    import plottool as pt

    new_mplshow = True and qres is not None
    mplshowtop = False and qres is not None
    qtinspect = False and qres is not None

    if new_mplshow:
        from ibeis.viz.interact import interact_query_decision
        print('Showing matplotlib window')
        comp_aids_all = ut.get_list_column(choicetup.sorted_aids, 0)
        comp_aids     = comp_aids_all[0:min(3, len(comp_aids_all))]
        suggestx      = ut.listfind(ibs.get_annot_names(comp_aids), name)
        suggest_aid   = None if suggestx is None else comp_aids[suggestx]
        name_decision_callback = incinfo['name_decision_callback']
        qvi = interact_query_decision.QueryVerificationInteraction(ibs, qres,
                                                                   comp_aids,
                                                                   suggest_aid,
                                                                   decision_callback=name_decision_callback,
                                                                   progress_current=incinfo['count'],
                                                                   progress_total=incinfo['nTotal'],
                                                                   fnum=incinfo['fnum'])
        qvi.fig.show()
    if mplshowtop:
        import guitool
        fnum = 513
        pt.figure(fnum=fnum, pnum=(2, 3, 1), doclf=True, docla=True)
        fig = qres.ishow_top(ibs, name_scoring=True, fnum=fnum, in_image=False,
                             annot_mode=0, sidebyside=False, show_query=True)
        fig.show()
        #fig.canvas.raise_()
        #from plottool import fig_presenter
        #fig_presenter.bring_to_front(fig)
        newname = ibs.make_next_name()
        newname_prefix = 'New Name:\n'
        if name is None:
            name = newname_prefix + newname

        aid_list = ut.get_list_column(choicetup.sorted_aids, 0)
        name_options = ibs.get_annot_names(aid_list) + [newname_prefix + newname]
        msg = 'Decide on query name. System suggests; ' + str(name)
        title = 'name decision'
        options = name_options[::-1]
        user_chosen_name = guitool.user_option(None, msg, title, options)  # NOQA
        if user_chosen_name is None:
            raise AssertionError('User Canceled Query')
        user_chosen_name = user_chosen_name.replace(newname_prefix, '')
        # TODO: Make the old interface use the correct sorted_aids format
        #name_decision_callback(user_chosen_name)
    if qtinspect:
        print('Showing qt inspect window')
        qres_wgt = qres.qt_inspect_gui(ibs, name_scoring=True)
        qres_wgt.show()
        qres_wgt.raise_()
    if qreq_ is not None:
        if qreq_.normalizer is None:
            print('normalizer is None!!')
        else:
            qreq_.normalizer.visualize(update=False, fnum=2)

    # Prompt the user (this could be swaped out with a qt or web interface)
    #if qtinspect:
    #    qres_wgt.close()
    #return user_chosen_name


def wait_for_user_exemplar_decision(autoexemplar_msg, exemplar_decision,
                                    exemplar_condience, incinfo=None):
    r""" hooks into to some method of getting user input for exemplars

    TODO: really good interface

    Args:
        autoexemplar_msg (?):
        exemplar_decision (?):
        exemplar_condience (?):

    Returns:
        ?: True

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-get_user_exemplar_decision

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis  # NOQA
        >>> # build test data
        >>> autoexemplar_msg = '?'
        >>> exemplar_decision = '?'
        >>> exemplar_condience = '?'
        >>> True = get_user_exemplar_decision(autoexemplar_msg, exemplar_decision, exemplar_condience)
        >>> # verify results
        >>> result = str(True)
        >>> print(result)
    """
    import guitool
    options = ['No', 'Yes']
    msg = 'Decide if exemplar. System suggests; ' + options[exemplar_decision]
    title = 'exemplar decision'
    responce = guitool.user_option(None, msg, title, options)  # NOQA
    if responce is None:
        raise AssertionError('User Canceled Query')
    if responce == 'Yes':
        return True
    elif responce == 'No':
        return False
    else:
        return None
    # TODO CALLBACK HERE
