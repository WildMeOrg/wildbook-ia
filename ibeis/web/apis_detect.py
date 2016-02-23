# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from ibeis.control import accessor_decors, controller_inject
from ibeis import constants as const
import utool as ut


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_api   = controller_inject.get_ibeis_flask_api(__name__)


@register_ibs_method
@accessor_decors.default_decorator
@accessor_decors.getter_1to1
@register_api('/api/detect/random_forest/', methods=['PUT', 'GET'])
def detect_random_forest(ibs, gid_list, species, **kwargs):
    """
    Runs animal detection in each image. Adds annotations to the database
    as they are found.

    Args:
        gid_list (list): list of image ids to run detection on
        species (str): string text of the species to identify

    Returns:
        aids_list (list): list of lists of annotation ids detected in each
            image

    CommandLine:
        python -m ibeis.control.IBEISControl --test-detect_random_forest --show

    RESTful:
        Method: PUT, GET
        URL:    /api/detect/random_forest/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> species = ibeis.const.TEST_SPECIES.ZEB_PLAIN
        >>> # execute function
        >>> aids_list = ibs.detect_random_forest(gid_list, species)
        >>> # Visualize results
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     from ibeis.viz import viz_image
        >>>     for fnum, gid in enumerate(gid_list):
        >>>         viz_image.show_image(ibs, gid, fnum=fnum)
        >>>     pt.show_if_requested()
        >>> # Remove newly detected annotations
        >>> ibs.delete_annots(ut.flatten(aids_list))
    """
    # TODO: Return confidence here as well
    print('[ibs] detecting using random forests')
    from ibeis.algo.detect import randomforest  # NOQA
    if isinstance(gid_list, int):
        gid_list = [gid_list]
    print('TYPE:' + str(type(gid_list)))
    print('GID_LIST:' + ut.truncate_str(str(gid_list)))
    detect_gen = randomforest.detect_gid_list_with_species(
        ibs, gid_list, species, **kwargs)
    # ibs.cfg.other_cfg.ensure_attr('detect_add_after', 1)
    # ADD_AFTER_THRESHOLD = ibs.cfg.other_cfg.detect_add_after
    print('TYPE:' + str(type(detect_gen)))
    aids_list = []
    for gid, (gpath, result_list) in zip(gid_list, detect_gen):
        aids = []
        for result in result_list:
            # Ideally, species will come from the detector with confidences
            # that actually mean something
            bbox = (result['xtl'], result['ytl'],
                    result['width'], result['height'])
            (aid,) = ibs.add_annots(
                [gid], [bbox], notes_list=['rfdetect'],
                species_list=[species], quiet_delete_thumbs=True,
                detect_confidence_list=[result['confidence']],
                skip_cleaning=True)
            aids.append(aid)
        aids_list.append(aids)
    ibs._clean_species()
    return aids_list


@register_ibs_method
@accessor_decors.default_decorator
@accessor_decors.getter_1to1
@register_api('/api/detect/cnn/yolo/', methods=['PUT', 'GET'])
def detect_cnn_yolo(ibs, gid_list, **kwargs):
    """
    Runs animal detection in each image. Adds annotations to the database
    as they are found.

    Args:
        gid_list (list): list of image ids to run detection on

    Returns:
        aids_list (list): list of lists of annotation ids detected in each
            image

    CommandLine:
        python -m ibeis.control.IBEISControl --test-detect_cnn_yolo --show

    RESTful:
        Method: PUT, GET
        URL:    /api/detect/cnn/yolo/

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.IBEISControl import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> gid_list = ibs.get_valid_gids()[0:2]
        >>> # execute function
        >>> aids_list = ibs.detect_cnn_yolo(gid_list)
        >>> # Visualize results
        >>> if ut.show_was_requested():
        >>>     import plottool as pt
        >>>     from ibeis.viz import viz_image
        >>>     for fnum, gid in enumerate(gid_list):
        >>>         viz_image.show_image(ibs, gid, fnum=fnum)
        >>>     pt.show_if_requested()
        >>> # Remove newly detected annotations
        >>> ibs.delete_annots(ut.flatten(aids_list))
    """
    # TODO: Return confidence here as well
    print('[ibs] detecting using CNN YOLO')
    from ibeis.algo.detect import yolo  # NOQA
    if isinstance(gid_list, int):
        gid_list = [gid_list]
    print('TYPE:' + str(type(gid_list)))
    print('GID_LIST:' + ut.truncate_str(str(gid_list)))
    detect_gen = yolo.detect_gid_list(ibs, gid_list, **kwargs)
    # ibs.cfg.other_cfg.ensure_attr('detect_add_after', 1)
    # ADD_AFTER_THRESHOLD = ibs.cfg.other_cfg.detect_add_after
    print('TYPE:' + str(type(detect_gen)))
    aids_list = []
    for gid, (gpath, result_list) in zip(gid_list, detect_gen):
        aids = []
        for result in result_list:
            bbox = (result['xtl'], result['ytl'],
                    result['width'], result['height'])
            (aid,) = ibs.add_annots(
                [gid], [bbox], notes_list=['cnnyolodetect'],
                species_list=[result['class']], quiet_delete_thumbs=True,
                detect_confidence_list=[result['confidence']],
                skip_cleaning=True)
            aids.append(aid)
        aids_list.append(aids)
    ibs._clean_species()
    return aids_list


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/detect/species/enabled/', methods=['GET'])
def has_species_detector(ibs, species_text):
    """
    TODO: extend to use non-constant species

    RESTful:
        Method: GET
        URL:    /api/detect/species/enabled/
    """
    # FIXME: infer this
    return species_text in const.SPECIES_WITH_DETECTORS


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/detect/species/', methods=['GET'])
def get_species_with_detectors(ibs):
    """
    RESTful:
        Method: GET
        URL:    /api/detect/species/
    """
    # FIXME: infer this
    return const.SPECIES_WITH_DETECTORS


@register_ibs_method
@accessor_decors.default_decorator
@register_api('/api/detect/species/working/', methods=['GET'])
def get_working_species(ibs):
    """
    RESTful:
        Method: GET
        URL:    /api/detect/species/working/
    """
    RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS = ut.get_argflag('--no-allspecies')

    species_nice_list = ibs.get_all_species_nice()
    species_text_list = ibs.get_all_species_texts()
    species_tup_list = zip(species_nice_list, species_text_list)
    if RESTRICT_TO_ONLY_SPECIES_WITH_DETECTORS:
        working_species_tups = [
            species_tup
            for species_tup in species_tup_list
            if ibs.has_species_detector(species_tup[1])
        ]
    else:
        working_species_tups = species_tup_list
    return working_species_tups

if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
