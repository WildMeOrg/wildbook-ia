from __future__ import absolute_import, division, print_function
import utool as ut
import six  # NOQA
import numpy as np  # NOQA
from vtool import keypoint as ktool  # NOQA
from vtool import spatial_verification as sver  # NOQA
from vtool import constrained_matching
"""
Todo tomorrow:

add coverage as option to IBEIS
add spatially constrained matching as option to IBEIS

"""


def param_interaction():
    r"""
    CommandLine:
        python -m vtool.test_constrained_matching --test-param_interaction

    Notes:
        python -m vtool.test_constrained_matching --test-param_interaction
        setparam normalizer_mode=nearby
        setparam normalizer_mode=far
        setparam ratio_thresh=.625
        setparam ratio_thresh=.5

        setparam ratio_thresh2=.625
        normalizer_mode=plus

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.test_constrained_matching import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> testtup = param_interaction()
        >>> # verify results
        >>> result = str(testtup)
        >>> print(result)
    """
    import plottool as pt
    USE_IBEIS = False and ut.is_developer()
    if USE_IBEIS:
        from ibeis.model.hots import devcases
        index = 2
        fpath1, fpath2, fpath3 = devcases.get_dev_test_fpaths(index)
        testtup1 = testdata_matcher(fpath1, fpath2)
        testtup2 = testdata_matcher(fpath1, fpath3)
    else:
        testtup1 = testdata_matcher('easy1.png', 'easy2.png')
        testtup2 = testdata_matcher('easy1.png', 'hard3.png')
    testtup_list = [testtup1, testtup2]
    simp_list = [SimpleMatcher(testtup) for testtup in testtup_list]
    varied_dict = dict([
        ('sver_xy_thresh', .1),
        ('ratio_thresh', .625),
        ('search_K', 7),
        ('ratio_thresh2', .625),
        ('sver_xy_thresh2', .01),
        ('normalizer_mode', ['nearby', 'far', 'plus'][1]),
        ('match_xy_thresh', .1),
    ])
    cfgdict_list = ut.all_dict_combinations(varied_dict)
    tried_configs = []

    # DEFINE CUSTOM INTRACTIONS
    custom_actions, valid_vizmodes, viz_index_, offset_fnum_ = make_custom_interactions(simp_list)
    # /DEFINE CUSTOM INTRACTIONS

    for cfgdict in ut.InteractiveIter(cfgdict_list,
                                      #default_action='reload',
                                      custom_actions=custom_actions,
                                      wraparound=True):
        for simp in simp_list:
            simp.run_matching(cfgdict=cfgdict)
        vizkey = valid_vizmodes[viz_index_[0]].replace('visualize_', '')
        print('vizkey = %r' % (vizkey,))
        for fnum_, simp in enumerate(simp_list):
            fnum = fnum_ + offset_fnum_[0]
            simp.visualize(vizkey, fnum=fnum)
        tried_configs.append(cfgdict.copy())
        print('Current Config = ')
        print(ut.dict_str(cfgdict))
        pt.present()
        pt.update()


def make_custom_interactions(simp_list):
    valid_vizmodes = ut.filter_startswith(dir(SimpleMatcher), 'visualize_')
    viz_index_ = [valid_vizmodes.index('visualize_matches')]
    def toggle_vizmode(iiter, actionkey, value, viz_index_=viz_index_):
        viz_index_[0] = (viz_index_[0] + 1) % len(valid_vizmodes)
        print('toggling')

    def set_param(iiter, actionkey, value, viz_index_=viz_index_):
        """
        value = 'search_K=3'
        """
        paramkey, paramval = value.split('=')
        print('parsing value=%r' % (value,))
        def strip_quotes(str_):
            dq = ut.DOUBLE_QUOTE
            sq = ut.SINGLE_QUOTE
            return str_.strip(dq).strip(sq).strip(dq)
        # Sanatize
        paramkey = strip_quotes(paramkey.strip())
        paramval = ut.smart_cast2(strip_quotes(paramval.strip()))
        print('setting cfgdict[%r]=%r' % (paramkey, paramval))
        iiter.iterable[iiter.index][paramkey] = paramval

    offset_fnum_ = [0]
    def offset_fnum(iiter, actionkey, value, offset_fnum_=offset_fnum_):
        offset_fnum_[0] += len(simp_list)

    custom_actions = [
        ('toggle', ['t'], 'toggles between ' + ut.cond_phrase(valid_vizmodes, 'and'), toggle_vizmode),
        ('offset_fnum', ['offset_fnum', 'o'], 'offset the figure number (keeps old figures)', offset_fnum),
        ('set_param', ['setparam', 's'], 'sets a config param using key=val format.  eg: setparam ratio_thresh=.1', set_param),
    ]
    return custom_actions, valid_vizmodes, viz_index_, offset_fnum_


def testdata_matcher(fname1='easy1.png', fname2='easy2.png'):
    """"
    fname1 = 'easy1.png'
    fname2 = 'hard3.png'
    """
    import utool as ut
    from vtool import image as gtool
    from vtool import features as feattool
    fpath1 = ut.grab_test_imgpath(fname1)
    fpath2 = ut.grab_test_imgpath(fname2)
    kpts1, vecs1 = feattool.extract_features(fpath1, rotation_invariance=True)
    #ut.embed()
    kpts2, vecs2 = feattool.extract_features(fpath2, rotation_invariance=True)
    rchip1 = gtool.imread(fpath1)
    rchip2 = gtool.imread(fpath2)
    #chip1_shape = vt.gtool.open_image_size(fpath1)
    chip2_shape = gtool.open_image_size(fpath2)
    dlen_sqrd2 = chip2_shape[0] ** 2 + chip2_shape[1]
    testtup = (rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2)
    return testtup


class SimpleMatcher(object):
    def __init__(simp, testtup):
        simp.testtup = None
        simp.basetup = None
        simp.nexttup = None
        if testtup is not None:
            simp.load_data(testtup)

    def load_data(simp, testtup):
        simp.testtup = testtup

    def run_matching(simp, testtup=None, cfgdict={}):
        if testtup is None:
            testtup = simp.testtup
        basetup, base_meta = constrained_matching.baseline_vsone_ratio_matcher(testtup, cfgdict)
        nexttup, next_meta = constrained_matching.spatially_constrianed_matcher(testtup, basetup, cfgdict)
        simp.nexttup = nexttup
        simp.basetup = basetup
        simp.testtup = testtup
        simp.base_meta = base_meta
        simp.next_meta = next_meta

    def setstate_testdata(simp):
        testtup = testdata_matcher()
        simp.run_matching(testtup)

    def visualize(simp, key, **kwargs):
        visualize_method = getattr(simp, 'visualize_' + key)
        return visualize_method(**kwargs)

    def start_new_viz(simp, nRows, nCols, fnum=None):
        import plottool as pt
        rchip1, rchip2, kpts1, vecs1, kpts2, vecs2, dlen_sqrd2  = simp.testtup
        fm_ORIG, fs_ORIG, fm_RAT, fs_RAT, fm_SV, fs_SV, H_RAT   = simp.basetup
        fm_SC, fs_SC, fm_SCR, fs_SCR, fm_SCRSV, fs_SCRSV, H_SCR = simp.nexttup
        fm_norm_RAT, fm_norm_SV                                 = simp.base_meta
        fm_norm_SC, fm_norm_SCR, fm_norm_SVSCR                  = simp.next_meta

        locals_ = ut.delete_dict_keys(locals(), ['title'])

        keytitle_tups = [
            ('ORIG', 'initial neighbors'),
            ('RAT', 'ratio filtered'),
            ('SV', 'ratio filtered + SV'),
            ('SC', 'spatially constrained'),
            ('SCR', 'spatially constrained + ratio'),
            ('SCRSV', 'spatially constrained + SV'),
        ]
        keytitle_dict = dict(keytitle_tups)
        key_list = ut.get_list_column(keytitle_tups, 0)
        matchtup_dict = {
            key: (locals_['fm_' + key], locals_['fs_' + key])
            for key in key_list
        }
        normtup_dict = {
            key: locals_.get('fm_norm_' + key, None)
            for key in key_list
        }

        next_pnum = pt.make_pnum_nextgen(nRows=nRows, nCols=nCols)
        if fnum is None:
            fnum = pt.next_fnum()
        INTERACTIVE = True
        if INTERACTIVE:
            from plottool import interact_helpers as ih
            fig = ih.begin_interaction('qres', fnum)
            ih.connect_callback(fig, 'button_press_event', on_single_match_clicked)
        else:
            pt.figure(fnum=fnum, doclf=True, docla=True)

        def show_matches_(key, **kwargs):
            assert key in key_list, 'unknown key=%r' % (key,)
            showkw = locals_.copy()
            pnum = next_pnum()
            showkw['pnum'] = pnum
            showkw['fnum'] = fnum
            showkw.update(kwargs)
            _fm, _fs = matchtup_dict[key]
            title = keytitle_dict[key]
            if kwargs.get('coverage'):
                from vtool import coverage_image
                kpts2, rchip2 = ut.dict_get(locals_, ('kpts2', 'rchip2'))
                kpts2_m = kpts2.take(_fm.T[1], axis=0)
                chip_shape2 = rchip2.shape
                coverage_mask, patch = coverage_image.make_coverage_mask(kpts2_m, chip_shape2, fx2_score=_fs)
                pt.imshow(coverage_mask * 255, pnum=pnum, fnum=fnum)
            else:
                if kwargs.get('norm', False):
                    _fm = normtup_dict[key]
                    assert _fm is not None, key
                    showkw['cmap'] = 'cool'
                    title += ' normalizers'
                show_matches(_fm, _fs, title=title, key=key, **showkw)
        # state hack
        #show_matches_.next_pnum = next_pnum
        return show_matches_

    def visualize_matches(simp, **kwargs):
        r"""
        CommandLine:
            python -m vtool.test_constrained_matching --test-visualize_matches --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.test_constrained_matching import *  # NOQA
            >>> import plottool as pt
            >>> simp = SimpleMatcher(testdata_matcher())
            >>> simp.run_matching()
            >>> result = simp.visualize_matches()
            >>> pt.show_if_requested()
        """
        nRows = 2
        nCols = 3
        show_matches_ = simp.start_new_viz(nRows, nCols, **kwargs)

        show_matches_('ORIG')
        show_matches_('RAT')
        show_matches_('SV')
        show_matches_('SC')
        show_matches_('SCR')
        show_matches_('SCRSV')

    def visualize_normalizers(simp, **kwargs):
        """
        CommandLine:
            python -m vtool.test_constrained_matching --test-visualize_normalizers --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.test_constrained_matching import *  # NOQA
            >>> import plottool as pt
            >>> simp = SimpleMatcher(testdata_matcher())
            >>> simp.run_matching()
            >>> result = simp.visualize_normalizers()
            >>> pt.show_if_requested()
        """
        nRows = 2
        nCols = 2
        show_matches_ = simp.start_new_viz(nRows, nCols, **kwargs)

        show_matches_('RAT')
        show_matches_('SCR')

        show_matches_('RAT', norm=True)
        show_matches_('SCR', norm=True)

        #show_matches_(fm_RAT, fs_RAT, title='ratio filtered')
        #show_matches_(fm_SCR, fs_SCR, title='constrained matches')

        #show_matches_(fm_norm_RAT, fs_RAT, title='ratio normalizers', cmap='cool')
        #show_matches_(fm_norm_SCR, fs_SCR, title='constrained normalizers', cmap='cool')

    def visualize_coverage(simp, **kwargs):
        """
        CommandLine:
            python -m vtool.test_constrained_matching --test-visualize_coverage --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from vtool.test_constrained_matching import *  # NOQA
            >>> import plottool as pt
            >>> simp = SimpleMatcher(testdata_matcher())
            >>> simp.run_matching()
            >>> result = simp.visualize_coverage()
            >>> pt.show_if_requested()
        """
        nRows = 2
        nCols = 2
        show_matches_ = simp.start_new_viz(nRows, nCols, **kwargs)

        show_matches_('SV', draw_lines=False)
        show_matches_('SCRSV', draw_lines=False)
        show_matches_('SV', coverage=True)
        show_matches_('SCRSV', coverage=True)


def show_matches(fm, fs, fnum=1, pnum=None, title='', key=None, simp=None,
                 cmap='hot', draw_lines=True, **locals_):
    #locals_ = locals()
    import plottool as pt
    from plottool import plot_helpers as ph
    # hack keys out of namespace
    keys = 'rchip1, rchip2, kpts1, kpts2'.split(', ')
    rchip1, rchip2, kpts1, kpts2 = ut.dict_take(locals_, keys)
    pt.figure(fnum=fnum, pnum=pnum)
    #doclf=True, docla=True)
    ax, xywh1, xywh2 = pt.show_chipmatch2(rchip1, rchip2, kpts1, kpts2, fm=fm,
                                          fs=fs, fnum=fnum, cmap=cmap,
                                          draw_lines=draw_lines, ori=True)
    ph.set_plotdat(ax, 'viztype', 'matches')
    ph.set_plotdat(ax, 'simp', simp)
    ph.set_plotdat(ax, 'key', key)
    title = title + '\n num=%d, sum=%.2f' % (len(fm), sum(fs))
    pt.set_title(title)
    return ax, xywh1, xywh2
    #pt.set_figtitle(title)
    # if update:
    #pt.iup()


#def ishow_matches(fm, fs, fnum=1, pnum=None, title='', cmap='hot', **locals_):
#    # TODO make things clickable
def on_single_match_clicked(event):
    from plottool import interact_helpers as ih
    from plottool import plot_helpers as ph
    """ result interaction mpl event callback slot """
    print('[viz] clicked result')
    if ih.clicked_outside_axis(event):
        pass
    else:
        ax = event.inaxes
        viztype = ph.get_plotdat(ax, 'viztype', '')
        #printDBG(str(event.__dict__))
        # Clicked a specific matches
        if viztype.startswith('matches'):
            #aid2 = ph.get_plotdat(ax, 'aid2', None)
            # Ctrl-Click
            evkey = '' if event.key is None else event.key
            simp = ph.get_plotdat(ax, 'simp', None)
            key = ph.get_plotdat(ax, 'key', None)
            print('evkey = %r' % evkey)
            if evkey.find('control') == 0:
                print('[viz] result control clicked')
                pass
            # Left-Click
            else:
                print(simp)
                print(key)
                print('[viz] result clicked')
                pass
    ph.draw()


def show_example():
    r"""
    CommandLine:
        python -m vtool.test_constrained_matching --test-show_example --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.test_constrained_matching import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> # execute function
        >>> result = show_example()
        >>> # verify results
        >>> print(result)
        >>> pt.present()
        >>> pt.show_if_requested()
    """
    #ut.util_grabdata.get_valid_test_imgkeys()
    testtup1 = testdata_matcher('easy1.png', 'easy2.png')
    testtup2 = testdata_matcher('easy1.png', 'hard3.png')
    simp1 = SimpleMatcher(testtup1)
    simp2 = SimpleMatcher(testtup2)
    simp1.run_matching()
    simp2.run_matching()
    #simp1.visualize_matches()
    #simp2.visualize_matches()
    simp1.visualize_normalizers()
    simp2.visualize_normalizers()
    #simp1.param_interaction()


if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.test_constrained_matching
        python -m vtool.test_constrained_matching --allexamples
        python -m vtool.test_constrained_matching --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
