# -*- coding: utf-8 -*-
"""
developer convenience functions for ibs

TODO: need to split up into sub modules:
    consistency_checks
    feasibility_fixes
    move the export stuff to dbio

    python -m utool.util_inspect check_module_usage --pat="ibsfuncs.py"

    then there are also convineience functions that need to be ordered at least
    within this file
"""
import logging
import utool as ut
from wbia.control import controller_inject
from wbia import annotmatch_funcs  # NOQA
import pytz


PST = pytz.timezone('US/Pacific')


# Inject utool function
(print, rrr, profile) = ut.inject2(__name__, '[research]')
logger = logging.getLogger('wbia')


# Must import class before injection
CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)


register_api = controller_inject.get_wbia_flask_api(__name__)


@register_ibs_method
def research_print_metrics(ibs, tag='metrics'):
    imageset_rowid_list = ibs.get_valid_imgsetids(is_special=False)
    imageset_text_list = ibs.get_imageset_text(imageset_rowid_list)

    global_gid_list = []
    global_cid_list = []
    for imageset_rowid, imageset_text in zip(imageset_rowid_list, imageset_text_list):
        imageset_text_ = imageset_text.strip().split(',')
        if len(imageset_text_) == 3:
            ggr, car, person = imageset_text_
            if ggr in ['GGR', 'GGR2']:
                gid_list = ibs.get_imageset_gids(imageset_rowid)
                global_gid_list += gid_list
                cid = ibs.add_contributors([imageset_text])[0]
                global_cid_list += [cid] * len(gid_list)

    assert len(global_gid_list) == len(set(global_gid_list))

    ibs.set_image_contributor_rowid(global_gid_list, global_cid_list)

    ######

    aid_list = ibs.get_valid_aids()

    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    quality_list = ibs.get_annot_qualities(aid_list)

    aids = []
    zipped = list(zip(aid_list, species_list, viewpoint_list, quality_list))
    for aid, species_, viewpoint_, quality_ in zipped:
        assert None not in [species_, viewpoint_, quality_]
        species_ = species_.lower()
        viewpoint_ = viewpoint_.lower()
        quality_ = int(quality_)
        if species_ != 'zebra_grevys':
            continue
        if 'right' not in viewpoint_:
            continue
        aids.append(aid)

    config = {
        'classifier_algo': 'densenet',
        'classifier_weight_filepath': 'canonical_zebra_grevys_v4',
    }
    prediction_list = ibs.depc_annot.get_property(
        'classifier', aids, 'class', config=config
    )
    confidence_list = ibs.depc_annot.get_property(
        'classifier', aids, 'score', config=config
    )
    confidence_list = [
        confidence if prediction == 'positive' else 1.0 - confidence
        for prediction, confidence in zip(prediction_list, confidence_list)
    ]
    flags = [confidence >= 0.5 for confidence in confidence_list]
    ibs.set_annot_canonical(aids, flags)

    ibs.print_dbinfo(with_ggr=True, with_map=True)
