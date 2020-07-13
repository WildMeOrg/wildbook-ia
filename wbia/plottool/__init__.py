# -*- coding: utf-8 -*-
# flake8: noqa
"""
Wrappers around matplotlib
"""
from __future__ import absolute_import, division, print_function

__version__ = '2.1.2'

import utool as ut

ut.noinject(__name__, '[plottool.__init__]')


# Hopefully this was imported sooner. TODO remove dependency
# from wbia.guitool import __PYQT__
# import wbia.guitool.__PYQT__ as __PYQT__
from . import __MPL_INIT__

__MPL_INIT__.init_matplotlib()

import matplotlib as mpl

# mpl.use('Qt4Agg')
import matplotlib.pyplot as plt

from . import plot_helpers as ph
from . import plot_helpers
from . import mpl_keypoint
from . import mpl_keypoint as mpl_kp
from . import mpl_sift as mpl_sift
from . import draw_func2
from . import draw_func2 as df2
from . import fig_presenter
from . import custom_constants
from . import custom_figure
from . import draw_sv
from . import viz_featrow
from . import viz_keypoints
from . import viz_image2
from . import plots
from . import interact_annotations
from . import interact_keypoints
from . import interact_multi_image
from . import interactions
from . import interact_impaint
from . import color_funcs

# from . import abstract_iteraction


# TODO utoolify this
IMPORT_TUPLES = [
    ('plot_helpers', None),
    ('fig_presenter', None),
    ('custom_constants', None),
    ('custom_figure', None),
    ('plots', None),
    ('draw_func2', None),
    ('interact_impaint', None),
    ('interactions', None),
    ('interact_multi_image', None),
    ('interact_keypoints', None),
    ('interact_matches', None),
    # ('abstract_iteraction', None),
    ('nx_helpers', None),
]

# The other module shouldn't exist.
# Functions in it need to be organized
from .plots import draw_hist_subbin_maxima

# from .draw_func2 import *  # NOQA
from .mpl_keypoint import draw_keypoints
from .mpl_sift import draw_sifts, render_sift_on_patch
from . import fig_presenter

import utool

# def reload_subs():
#    rrr()
#    df2.rrr()
#    plot_helpers.rrr()
#    draw_sv.rrr()
#    viz_keypoints.rrr()
#    viz_image2.rrr()
#    rrr()

# rrrr = reload_subs


import sys

__DYNAMIC__ = '--nodyn' not in sys.argv

# __DYNAMIC__ = '--dyn' in sys.argv
"""
python -c "import wbia.plottool" --dump-plottool-init
python -c "import wbia.plottool" --update-plottool-init
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
    import_execstr = util_importer.dynamic_import(
        __name__, IMPORT_TUPLES, ignore_endswith=ignore_endswith
    )
    exec(import_execstr)
    DOELSE = False
else:
    # Do the nonexec import (can force it to happen no matter what if alwyas set
    # to True)
    DOELSE = True

if DOELSE:
    pass
    # <AUTOGEN_INIT>

    from . import plot_helpers
    from . import fig_presenter
    from . import custom_constants
    from . import custom_figure
    from . import plots
    from . import draw_func2
    from . import interact_impaint
    from . import interactions
    from . import interact_multi_image
    from . import interact_keypoints
    from . import interact_matches
    from . import nx_helpers
    from .plot_helpers import (
        SIFT_OR_VECFIELD,
        del_plotdat,
        draw,
        ensureqt,
        get_bbox_centers,
        get_plotdat,
        get_plotdat_dict,
        get_square_row_cols,
        kp_info,
        qt4ensure,
        set_plotdat,
    )
    from .fig_presenter import (
        SLEEP_TIME,
        VERBOSE,
        all_figures_bring_to_front,
        all_figures_show,
        all_figures_tight_layout,
        all_figures_tile,
        bring_to_front,
        close_all_figures,
        close_figure,
        get_all_figures,
        get_all_qt4_wins,
        get_all_windows,
        get_fig,
        get_figure_window,
        get_geometry,
        get_main_win_base,
        iup,
        iupdate,
        present,
        register_qt4_win,
        reset,
        set_geometry,
        show,
        show_figure,
        unregister_qt4_win,
        update,
    )
    from .custom_constants import (
        BLACK,
        BLUE,
        BRIGHT_GREEN,
        BRIGHT_PURPLE,
        DARK_BLUE,
        DARK_GREEN,
        DARK_ORANGE,
        DARK_RED,
        DARK_YELLOW,
        DEEP_PINK,
        DPI,
        FALSE_RED,
        FIGSIZE,
        FIGSIZE_BIGGER,
        FIGSIZE_GOLD,
        FIGSIZE_HUGE,
        FIGSIZE_MED,
        FIGSIZE_SQUARE,
        FONTS,
        FontProp,
        GRAY,
        GREEN,
        LARGE,
        LARGER,
        LIGHTGRAY,
        LIGHT_BLUE,
        LIGHT_GREEN,
        LIGHT_PINK,
        LIGHT_PURPLE,
        MED,
        NEUTRAL,
        NEUTRAL_BLUE,
        ORANGE,
        PHI,
        PHI_denom,
        PHI_numer,
        PINK,
        PURPLE,
        PURPLE2,
        RED,
        SMALL,
        SMALLER,
        SMALLEST,
        TRUE_BLUE,
        TRUE_GREEN,
        UNKNOWN_PURP,
        WHITE,
        YELLOW,
        golden_wh,
        golden_wh2,
    )
    from .custom_figure import (
        FIGTITLE_SIZE,
        LABEL_SIZE,
        LEGEND_SIZE,
        TITLE_SIZE,
        cla,
        clf,
        customize_figure,
        customize_fontprop,
        figure,
        gca,
        gcf,
        get_ax,
        get_image_from_figure,
        prepare_figure_for_save,
        prepare_figure_fpath,
        sanitize_img_ext,
        sanitize_img_fname,
        save_figure,
        set_figtitle,
        set_ticks,
        set_title,
        set_xlabel,
        set_xticks,
        set_ylabel,
        set_yticks,
        split,
    )
    from .plots import (
        colorline,
        draw_histogram,
        draw_time_distribution,
        draw_time_histogram,
        draw_timedelta_pie,
        estimate_pdf,
        get_good_logyscale_kwargs,
        interval_line_plot,
        interval_stats_plot,
        is_default_dark_bg,
        multi_plot,
        plot_densities,
        plot_multiple_scores,
        plot_pdf,
        plot_probabilities,
        plot_probs,
        plot_rank_cumhist,
        plot_score_histograms,
        plot_search_surface,
        plot_sorted_scores,
        plot_stems,
        set_logyscale_from_data,
        unicode_literals,
        word_histogram2,
        wordcloud,
        zoom_effect01,
    )
    from .draw_func2 import (
        BASE_FNUM,
        DARKEN,
        DEBUG,
        DF2_DIVIDER_KEY,
        FALSE,
        LEGEND_LOCATION,
        OffsetImage2,
        RenderingContext,
        SAFE_POS,
        TAU,
        TMP_mevent,
        TRUE,
        absolute_lbl,
        add_alpha,
        adjust_subplots,
        adjust_subplots,
        adjust_subplots_safe,
        append_phantom_legend_label,
        ax_absolute_text,
        ax_relative_text,
        axes_bottom_button_bar,
        cartoon_stacked_rects,
        color_orimag,
        color_orimag_colorbar,
        colorbar,
        customize_colormap,
        dark_background,
        distinct_colors,
        distinct_markers,
        draw_bbox,
        draw_border,
        draw_boxedX,
        draw_keypoint_gradient_orientations,
        draw_keypoint_patch,
        draw_kpts2,
        draw_line_segments,
        draw_line_segments2,
        draw_lines2,
        draw_patches_and_sifts,
        draw_stems,
        draw_text,
        draw_text_annotations,
        draw_vector_field,
        ensure_divider,
        ensure_fnum,
        execstr_global,
        extract_axes_extents,
        fig_relative_text,
        fnum_generator,
        get_all_markers,
        get_axis_bbox,
        get_axis_xy_width_height,
        get_binary_svm_cmap,
        get_num_rc,
        get_orientation_color,
        get_pnum_func,
        get_save_directions,
        imshow,
        imshow_null,
        is_texmode,
        label_to_colors,
        legend,
        lighten_rgb,
        lowerright_text,
        make_axes_locatable,
        make_bbox,
        make_bbox_positioners,
        make_fnum_nextgen,
        make_ori_legend_img,
        make_pnum_nextgen,
        next_fnum,
        overlay_icon,
        pad_axes,
        param_plot_iterator,
        parse_fontkw,
        plot,
        plot2,
        plotWidget,
        plot_bars,
        plot_descriptor_signature,
        plot_fmatch,
        plot_func,
        plot_hist,
        plot_histpdf,
        plot_sift_signature,
        plot_surface3d,
        pnum_generator,
        postsetup_axes,
        presetup_axes,
        print_valid_cmaps,
        remove_patches,
        render_figure_to_image,
        reverse_colormap,
        rotate_plot,
        scores_to_cmap,
        scores_to_color,
        set_axis_extent,
        set_axis_limit,
        set_figsize,
        show_chipmatch2,
        show_histogram,
        show_if_requested,
        show_kpts,
        show_phantom_legend_labels,
        show_signature,
        show_was_requested,
        small_xticks,
        small_yticks,
        space_xticks,
        space_yticks,
        to_base255,
        udpate_adjust_subplots,
        unique_rows,
        update_figsize,
        upperleft_text,
        upperright_text,
        variation_trunctate,
        width_from,
    )
    from .interact_impaint import (
        PAINTER_BASE,
        PaintInteraction,
        draw_demo,
        impaint_mask2,
    )
    from .interactions import (
        ExpandableInteraction,
        PanEvents,
        check_if_subinteract,
        pan_factory,
        zoom_factory,
    )
    from .interact_multi_image import (
        BASE_CLASS,
        Button,
        MultiImageInteraction,
    )
    from .interact_keypoints import (
        KeypointInteraction,
        draw_feat_row,
        ishow_keypoints,
        show_keypoints,
    )
    from .interact_matches import (
        MatchInteraction2,
        show_keypoint_gradient_orientations,
    )
    from .nx_helpers import (
        GraphVizLayoutConfig,
        LARGE_GRAPH,
        apply_graph_layout_attrs,
        draw_network2,
        dump_nx_ondisk,
        ensure_nonhex_color,
        format_anode_pos,
        get_explicit_graph,
        get_nx_layout,
        make_agraph,
        make_agraph_args,
        netx_draw_images_at_positions,
        nx_agraph_layout,
        parse_aedge_layout_attrs,
        parse_anode_layout_attrs,
        parse_point,
        show_nx,
    )
    import utool

    print, rrr, profile = utool.inject2(__name__, '[plottool]')

    def reassign_submodule_attributes(verbose=True):
        """
        why reloading all the modules doesnt do this I don't know
        """
        import sys

        if verbose and '--quiet' not in sys.argv:
            print('dev reimport')
        # Self import
        import wbia.plottool

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
        if verbose:
            print('Reloading submodules')
        rrr(verbose=verbose)

        def wrap_fbrrr(mod):
            def fbrrr(*args, **kwargs):
                """ fallback reload """
                if verbose:
                    print('No fallback relaod for mod=%r' % (mod,))
                # Breaks ut.Pref (which should be depricated anyway)
                # import imp
                # imp.reload(mod)

            return fbrrr

        def get_rrr(mod):
            if hasattr(mod, 'rrr'):
                return mod.rrr
            else:
                return wrap_fbrrr(mod)

        def get_reload_subs(mod):
            return getattr(mod, 'reload_subs', wrap_fbrrr(mod))

        get_rrr(plot_helpers)(verbose=verbose)
        get_rrr(fig_presenter)(verbose=verbose)
        get_rrr(custom_constants)(verbose=verbose)
        get_rrr(custom_figure)(verbose=verbose)
        get_rrr(plots)(verbose=verbose)
        get_rrr(draw_func2)(verbose=verbose)
        get_rrr(interact_impaint)(verbose=verbose)
        get_rrr(interactions)(verbose=verbose)
        get_rrr(interact_multi_image)(verbose=verbose)
        get_rrr(interact_keypoints)(verbose=verbose)
        get_rrr(interact_matches)(verbose=verbose)
        get_rrr(nx_helpers)(verbose=verbose)
        rrr(verbose=verbose)
        try:
            # hackish way of propogating up the new reloaded submodule attributes
            reassign_submodule_attributes(verbose=verbose)
        except Exception as ex:
            print(ex)

    rrrr = reload_subs
    # </AUTOGEN_INIT>
