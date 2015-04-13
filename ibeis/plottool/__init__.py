# flake8: noqa
from __future__ import absolute_import, division, print_function

__version__ = '1.0.0.dev1'

import utool as ut
ut.noinject(__name__, '[plottool.__init__]')


# Hopefully this was imported sooner. TODO remove dependency
from guitool import __PYQT__
from plottool import __MPL_INIT__
__MPL_INIT__.init_matplotlib()

import matplotlib as mpl
#mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

from plottool import plot_helpers as ph
from plottool import plot_helpers
from plottool import mpl_keypoint
from plottool import mpl_keypoint as mpl_kp
from plottool import mpl_sift as mpl_sift
from plottool import draw_func2
from plottool import draw_func2 as df2
from plottool import fig_presenter
from plottool import custom_constants
from plottool import custom_figure
from plottool import draw_sv
from plottool import viz_featrow
from plottool import viz_keypoints
from plottool import viz_image2
from plottool import plots
from plottool import interact_annotations
from plottool import interact_keypoints
from plottool import interact_multi_image


# TODO utoolify this
IMPORT_TUPLES = [
    ('plot_helpers', None),
    ('fig_presenter', None),
    ('custom_constants', None),
    ('custom_figure', None),
    ('plots', None),
    ('draw_func2', None),
]

# The other module shouldn't exist.
# Functions in it need to be organized
from plottool.plots import draw_hist_subbin_maxima
#from plottool.draw_func2 import *  # NOQA
from plottool.mpl_keypoint import draw_keypoints
from plottool.mpl_sift import draw_sifts
from plottool import fig_presenter

import utool
#print, print_, printDBG, rrr, profile = utool.inject(__name__, '[plottool]')

#def reload_subs():
#    rrr()
#    df2.rrr()
#    plot_helpers.rrr()
#    draw_sv.rrr()
#    viz_keypoints.rrr()
#    viz_image2.rrr()
#    rrr()

#rrrr = reload_subs


import sys
__DYNAMIC__ = not '--nodyn' in sys.argv

#__DYNAMIC__ = '--dyn' in sys.argv
"""
python -c "import plottool" --dump-plottool-init
python -c "import plottool" --update-plottool-init
"""


DOELSE = False
if __DYNAMIC__:
    # TODO: import all utool external prereqs. Then the imports will not import
    # anything that has already in a toplevel namespace
    # COMMENTED OUT FOR FROZEN __INIT__
    # Dynamically import listed util libraries and their members.
    from utool._internal import util_importer
    # FIXME: this might actually work with rrrr, but things arent being
    # reimported because they are already in the modules list
    ignore_endswith = ['_cyth']
    import_execstr = util_importer.dynamic_import(__name__, IMPORT_TUPLES, ignore_endswith=ignore_endswith)
    exec(import_execstr)
    DOELSE = False
else:
    # Do the nonexec import (can force it to happen no matter what if alwyas set
    # to True)
    DOELSE = True

if DOELSE:
    pass
    # <AUTOGEN_INIT>

    from plottool import plot_helpers
    from plottool import fig_presenter
    from plottool import custom_constants
    from plottool import custom_figure
    from plottool import plots
    from plottool import draw_func2
    from plottool.plot_helpers import (SIFT_OR_VECFIELD, draw, dump_figure, 
                                       get_bbox_centers, get_plotdat, 
                                       get_square_row_cols, kp_info, 
                                       set_plotdat,) 
    from plottool.fig_presenter import (Qt, SLEEP_TIME, 
                                        all_figures_bring_to_front, 
                                        all_figures_show, 
                                        all_figures_tight_layout, 
                                        all_figures_tile, bring_to_front, 
                                        close_all_figures, close_figure, 
                                        get_all_figures, get_all_qt4_wins, 
                                        get_all_windows, get_fig, get_geometry, 
                                        get_main_win_base, iup, iupdate, 
                                        present, register_qt4_win, reset, 
                                        set_geometry, show, unregister_qt4_win, 
                                        update,) 
    from plottool.custom_constants import (BLACK, BLUE, DARK_BLUE, DARK_GREEN, 
                                           DARK_ORANGE, DARK_RED, DARK_YELLOW, 
                                           DEEP_PINK, DPI, FALSE_RED, FIGSIZE, 
                                           FIGSIZE_BIGGER, FIGSIZE_GOLD, 
                                           FIGSIZE_HUGE, FIGSIZE_MED, 
                                           FIGSIZE_SQUARE, FONTS, FontProp, 
                                           GRAY, GREEN, LARGE, LIGHT_BLUE, MED, 
                                           ORANGE, PHI, PHI_denom, PHI_numer, 
                                           PINK, PURPLE, RED, SMALL, SMALLER, 
                                           SMALLEST, TRUE_BLUE, TRUE_GREEN, 
                                           UNKNOWN_PURP, WHITE, YELLOW, 
                                           golden_wh, golden_wh2,) 
    from plottool.custom_figure import (cla, clf, customize_figure, 
                                        customize_fontprop, figure, gca, gcf, 
                                        get_ax, prepare_figure_for_save, 
                                        prepare_figure_fpath, sanitize_img_ext, 
                                        sanitize_img_fname, save_figure, 
                                        set_figtitle, set_ticks, set_title, 
                                        set_xlabel, set_xticks, set_ylabel, 
                                        set_yticks, split,) 
    from plottool.plots import (colorline, estimate_pdf, 
                                get_good_logyscale_kwargs, interval_stats_plot, 
                                plot_densities, plot_pdf, plot_probabilities, 
                                plot_probs, plot_sorted_scores, plot_stems, 
                                set_logyscale_from_data,) 
    from plottool.draw_func2 import (BASE_FNUM, DARKEN, DEBUG, LEGEND_LOCATION, 
                                     LineCollection, SAFE_POS, TAU, TMP_mevent, 
                                     absolute_lbl, add_alpha, adjust_subplots, 
                                     adjust_subplots_safe, 
                                     adjust_subplots_xlabels, 
                                     adjust_subplots_xylabels, 
                                     append_phantom_legend_label, 
                                     ax_absolute_text, ax_relative_text, 
                                     axes_bottom_button_bar, color_orimag, 
                                     color_orimag_colorbar, colorbar, 
                                     customize_colormap, dark_background, 
                                     distinct_colors, draw_bbox, draw_border, 
                                     draw_boxedX, 
                                     draw_keypoint_gradient_orientations, 
                                     draw_keypoint_patch, draw_kpts2, 
                                     draw_lines2, draw_stems, draw_text, 
                                     draw_vector_field, ensure_divider, 
                                     ensure_fnum, execstr_global, 
                                     fig_relative_text, 
                                     get_axis_xy_width_height, 
                                     get_binary_svm_cmap, 
                                     get_orientation_color, get_pnum_func, 
                                     imshow, imshow_null, label_to_colors, 
                                     legend, lighten_rgb, lowerright_text, 
                                     make_axes_locatable, 
                                     make_bbox_positioners, 
                                     make_ori_legend_img, make_pnum_nextgen, 
                                     next_fnum, param_plot_iterator, plot, 
                                     plot2, plotWidget, plot_bars, plot_fmatch, 
                                     plot_hist, plot_histpdf, 
                                     plot_sift_signature, plot_surface3d, 
                                     pnum_generator, print_valid_cmaps, 
                                     remove_patches, reverse_colormap, 
                                     rotate_plot, scores_to_cmap, 
                                     scores_to_color, show_all_colormaps, 
                                     show_chipmatch2, show_histogram, 
                                     show_if_requested, 
                                     show_phantom_legend_labels, 
                                     show_signature, show_was_requested, 
                                     small_xticks, small_yticks, space_xticks, 
                                     space_yticks, stack_image_list, 
                                     stack_image_recurse, stack_images, 
                                     stack_square_images, to_base255, 
                                     unique_rows, upperleft_text, 
                                     upperright_text, variation_trunctate, 
                                     width_from,) 
    import utool
    print, print_, printDBG, rrr, profile = utool.inject(
        __name__, '[plottool]')
    
    
    def reassign_submodule_attributes(verbose=True):
        """
        why reloading all the modules doesnt do this I don't know
        """
        import sys
        if verbose and '--quiet' not in sys.argv:
            print('dev reimport')
        # Self import
        import plottool
        # Implicit reassignment.
        seen_ = set([])
        for tup in IMPORT_TUPLES:
            if len(tup) > 2 and tup[2]:
                continue  # dont import package names
            submodname, fromimports = tup[0:2]
            submod = getattr(plottool, submodname)
            for attr in dir(submod):
                if attr.startswith('_'):
                    continue
                if attr in seen_:
                    # This just holds off bad behavior
                    # but it does mimic normal util_import behavior
                    # which is good
                    continue
                seen_.add(attr)
                setattr(plottool, attr, getattr(submod, attr))
    
    
    def reload_subs(verbose=True):
        """ Reloads plottool and submodules """
        rrr(verbose=verbose)
        def fbrrr(*args, **kwargs):
            """ fallback reload """
            pass
        getattr(plot_helpers, 'rrr', fbrrr)(verbose=verbose)
        getattr(fig_presenter, 'rrr', fbrrr)(verbose=verbose)
        getattr(custom_constants, 'rrr', fbrrr)(verbose=verbose)
        getattr(custom_figure, 'rrr', fbrrr)(verbose=verbose)
        getattr(plots, 'rrr', fbrrr)(verbose=verbose)
        getattr(draw_func2, 'rrr', fbrrr)(verbose=verbose)
        rrr(verbose=verbose)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception as ex:
            print(ex)
    rrrr = reload_subs
    # </AUTOGEN_INIT>