from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_func2 as df2
import numpy as np
from plottool import plot_helpers as ph
#(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_keypoints]', DEBUG=False)
utool.noinject(__name__, '[viz_keypoints]')


def testdata_kpts():
    import utool as ut
    import vtool as vt
    import pyhesaff
    img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='star.png'))
    kwargs = ut.parse_dict_from_argv(pyhesaff.get_hesaff_default_params())
    (kpts, vecs) = pyhesaff.detect_kpts(img_fpath, **kwargs)
    imgBGR = vt.imread(img_fpath)
    return kpts, vecs, imgBGR


def show_keypoints(chip, kpts, fnum=0, pnum=None, **kwargs):
    #printDBG('[df2.show_kpts] %r' % (kwargs.keys(),))
    fig, ax = df2.imshow(chip, fnum=fnum, pnum=pnum, **kwargs)
    _annotate_kpts(kpts, **kwargs)
    ph.set_plotdat(ax, 'viztype', 'keypoints')
    ph.set_plotdat(ax, 'kpts', kpts)
    if kwargs.get('ddd', False):
        ph.draw()


#@utool.indent_func
def _annotate_kpts(kpts_, sel_fx=None, **kwargs):
    """
        Args:
            kpts_ (ndarray): keypoints
            sel_fx (None):

        Keywords:
            color:  3/4-tuple, ndarray, or str

        Returns:
            None

        Example:
            >>> from plottool.viz_keypoints import *  # NOQA
            >>> sel_fx = None
            >>> kpts = np.array([[  92.9246,   17.5453,    7.8103,   -3.4594,   10.8566,    0.    ],
            ...                  [  76.8585,   24.7918,   11.4412,   -3.2634,    9.6287,    0.    ],
            ...                  [ 140.6303,   24.9027,   10.4051,  -10.9452, 10.5991,    0.    ],])

    """
    '''
    python -c "import utool; utool.print_auto_docstr('plottool.viz_keypoints', '_annotate_kpts')"
    '''
    if len(kpts_) == 0:
        print('len(kpts_) == 0...')
        return
    color = kwargs.get('color', 'distinct' if sel_fx is None else df2.ORANGE)
    if color == 'distinct':
        # hack for distinct colors
        color = df2.distinct_colors(len(kpts_))  # , randomize=True)
    # Keypoint drawing kwargs
    drawkpts_kw = {
        'ell': True,
        'pts': False,
        'ell_alpha': .4,
        'ell_linewidth': 2,
        'ell_color': color,
    }
    drawkpts_kw.update(kwargs)

    # draw all keypoints
    if sel_fx is None:
        df2.draw_kpts2(kpts_, **drawkpts_kw)
    else:
        # dont draw the selected keypoint in this batch
        nonsel_kpts_ = np.vstack((kpts_[0:sel_fx], kpts_[sel_fx + 1:]))
        # Draw selected keypoint
        sel_kpts = kpts_[sel_fx:sel_fx + 1]
        import utool as ut
        if ut.isiterable(color) and ut.isiterable(color[0]):
            # hack for distinct colors
            drawkpts_kw['ell_color'] = color[0:sel_fx] + color[sel_fx + 1:]
        drawkpts_kw
        drawkpts_kw2 = drawkpts_kw.copy()
        drawkpts_kw2.update({
            'ell_color': df2.BLUE,
            'eig':  True,
            'rect': True,
            'ori':  True,
        })
        df2.draw_kpts2(nonsel_kpts_, **drawkpts_kw)
        df2.draw_kpts2(sel_kpts, **drawkpts_kw2)
