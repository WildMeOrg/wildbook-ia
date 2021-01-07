# -*- coding: utf-8 -*-
import logging

# from os.path import expanduser, join
from wbia import constants as const
from wbia.control.controller_inject import make_ibs_register_decorator
import utool as ut
import os
from collections import defaultdict

# illustration imports
from shutil import copy
from PIL import Image, ImageDraw
import wbia.plottool as pt


logger = logging.getLogger('wbia')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)

PARALLEL = not const.CONTAINERIZED
INPUT_SIZE = 224

INMEM_ASSIGNER_MODELS = {}

SPECIES_CONFIG_MAP = {
    'wild_dog': {
        'model_file': '/tmp/balanced_wd.joblib',
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'annot_feature_col': 'assigner_viewpoint_features',
    },
    'wild_dog_dark': {
        'model_file': '/tmp/balanced_wd.joblib',
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'annot_feature_col': 'assigner_viewpoint_features',
    },
    'wild_dog_light': {
        'model_file': '/tmp/balanced_wd.joblib',
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'annot_feature_col': 'assigner_viewpoint_features',
    },
    'wild_dog_puppy': {
        'model_file': '/tmp/balanced_wd.joblib',
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'annot_feature_col': 'assigner_viewpoint_features',
    },
    'wild_dog_standard': {
        'model_file': '/tmp/balanced_wd.joblib',
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'annot_feature_col': 'assigner_viewpoint_features',
    },
    'wild_dog_tan': {
        'model_file': '/tmp/balanced_wd.joblib',
        'model_url': 'https://wildbookiarepository.azureedge.net/models/assigner.wd_v0.joblib',
        'annot_feature_col': 'assigner_viewpoint_features',
    },
}


@register_ibs_method
def _are_part_annots(ibs, aid_list):
    r"""
    returns a boolean list representing if each aid in aid_list is a part annot.
    This determination is made by the presence of a '+' in the species.

    Args:
        ibs         (IBEISController): IBEIS / WBIA controller object
        aid_list  (int): annot ids to split

    CommandLine:
        python -m wbia.algo.detect.assigner _are_part_annots

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from wbia.algo.detect.assigner import *
        >>> from wbia.algo.detect.train_assigner import *
        >>> ibs = assigner_testdb_ibs()
        >>> aids = ibs.get_valid_aids()
        >>> result = ibs._are_part_annots(aids)
        >>> print(result)
        [False, False, True, True, False, True, False, True]
    """
    species = ibs.get_annot_species(aid_list)
    are_parts = ['+' in specie for specie in species]
    return are_parts


def all_part_pairs(ibs, gid_list):
    r"""
    Returns all possible part,body pairs from aids in gid_list, in the format of
    two parralel lists: the first being all parts, the second all bodies

    Args:
        ibs         (IBEISController): IBEIS / WBIA controller object
        gid_list  (int): gids in question

    CommandLine:
        python -m wbia.algo.detect.assigner _are_part_annots

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from wbia.algo.detect.assigner import *
        >>> from wbia.algo.detect.train_assigner import *
        >>> ibs = assigner_testdb_ibs()
        >>> gids = ibs.get_valid_gids()
        >>> all_part_pairs = all_part_pairs(ibs, gids)
        >>> parts = all_part_pairs[0]
        >>> bodies = all_part_pairs[1]
        >>> all_aids = ibs.get_image_aids(gids)
        >>> all_aids = [aid for aids in all_aids for aid in aids]  # flatten
        >>> assert (set(parts) & set(bodies)) == set({})
        >>> assert (set(parts) | set(bodies)) == set(all_aids)
        >>> result = all_part_pairs
        >>> print(result)
        ([3, 3, 4, 4, 6, 8], [1, 2, 1, 2, 5, 7])
    """
    all_aids = ibs.get_image_aids(gid_list)
    all_aids_are_parts = [ibs._are_part_annots(aids) for aids in all_aids]
    all_part_aids = [
        [aid for (aid, part) in zip(aids, are_parts) if part]
        for (aids, are_parts) in zip(all_aids, all_aids_are_parts)
    ]
    all_body_aids = [
        [aid for (aid, part) in zip(aids, are_parts) if not part]
        for (aids, are_parts) in zip(all_aids, all_aids_are_parts)
    ]
    part_body_parallel_lists = [
        _all_pairs_parallel(parts, bodies)
        for parts, bodies in zip(all_part_aids, all_body_aids)
    ]
    all_parts = [
        aid
        for part_body_parallel_list in part_body_parallel_lists
        for aid in part_body_parallel_list[0]
    ]
    all_bodies = [
        aid
        for part_body_parallel_list in part_body_parallel_lists
        for aid in part_body_parallel_list[1]
    ]
    return all_parts, all_bodies


def _all_pairs_parallel(list_a, list_b):
    # is tested by all_part_pairs above
    pairs = [(a, b) for a in list_a for b in list_b]
    pairs_a = [pair[0] for pair in pairs]
    pairs_b = [pair[1] for pair in pairs]
    return pairs_a, pairs_b


@register_ibs_method
def assign_parts(ibs, all_aids, cutoff_score=0.5):
    r"""
    Main assigner method; makes assignments on all_aids based on assigner scores.

    Args:
        ibs         (IBEISController): IBEIS / WBIA controller object
        aid_list  (int): aids in question
        cutoff_score: the threshold for the aids' assigner scores, under which no assignments are made

    Returns:
        tuple of two lists: all_assignments (a list of tuples, each tuple grouping
        aids assigned to a single animal), and all_unassigned_aids, which are the aids that did not meet the cutoff_score or whose body/part

    CommandLine:
        python -m wbia.algo.detect.assigner _are_part_annots

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from wbia.algo.detect.assigner import *
        >>> from wbia.algo.detect.train_assigner import *
        >>> ibs = assigner_testdb_ibs()
        >>> aids = ibs.get_valid_aids()
        >>> result = ibs.assign_parts(aids)
        >>> assigned_pairs = result[0]
        >>> unassigned_aids = result[1]
        >>> assigned_aids = [item for pair in assigned_pairs for item in pair]
        >>> #  no overlap between assigned and unassigned aids
        >>> assert (set(assigned_aids) & set(unassigned_aids) == set({}))
        >>> #  all aids are either assigned or unassigned
        >>> assert (set(assigned_aids) | set(unassigned_aids) == set(aids))
        >>> ([(3, 1), (6, 5), (8, 7)], [2, 4])
    """
    gids = ibs.get_annot_gids(all_aids)
    gid_to_aids = defaultdict(list)
    for gid, aid in zip(gids, all_aids):
        gid_to_aids[gid] += [aid]

    all_assignments = []
    all_unassigned_aids = []

    for gid in gid_to_aids.keys():
        this_pairs, this_unassigned = assign_parts_one_image(
            ibs, gid_to_aids[gid], cutoff_score
        )
        all_assignments += this_pairs
        all_unassigned_aids += this_unassigned

    return all_assignments, all_unassigned_aids


@register_ibs_method
def assign_parts_one_image(ibs, aid_list, cutoff_score=0.5):
    r"""
    Main assigner method; makes assignments on all_aids based on assigner scores.

    Args:
        ibs         (IBEISController): IBEIS / WBIA controller object
        aid_list  (int): aids in question
        cutoff_score: the threshold for the aids' assigner scores, under which no assignments are made

    Returns:
        tuple of two lists: all_assignments (a list of tuples, each tuple grouping
        aids assigned to a single animal), and all_unassigned_aids, which are the aids that did not meet the cutoff_score or whose body/part

    CommandLine:
        python -m wbia.algo.detect.assigner _are_part_annots

    Example:
        >>> # ENABLE_DOCTEST
        >>> import utool as ut
        >>> from wbia.algo.detect.assigner import *
        >>> from wbia.algo.detect.train_assigner import *
        >>> ibs = assigner_testdb_ibs()
        >>> gid = 1
        >>> aids = ibs.get_image_aids(gid)
        >>> result = ibs.assign_parts_one_image(aids)
        >>> assigned_pairs = result[0]
        >>> unassigned_aids = result[1]
        >>> assigned_aids = [item for pair in assigned_pairs for item in pair]
        >>> #  no overlap between assigned and unassigned aids
        >>> assert (set(assigned_aids) & set(unassigned_aids) == set({}))
        >>> #  all aids are either assigned or unassigned
        >>> assert (set(assigned_aids) | set(unassigned_aids) == set(aids))
        >>> ([(3, 1)], [2, 4])
    """
    all_species = ibs.get_annot_species(aid_list)
    # put unsupported species into the all_unassigned_aids list
    all_species_no_parts = [species.split('+')[0] for species in all_species]
    assign_flag_list = [
        species in SPECIES_CONFIG_MAP.keys() for species in all_species_no_parts
    ]

    unassigned_aids_noconfig = ut.filterfalse_items(aid_list, assign_flag_list)
    aid_list = ut.compress(aid_list, assign_flag_list)

    are_part_aids = _are_part_annots(ibs, aid_list)
    part_aids = ut.compress(aid_list, are_part_aids)
    body_aids = ut.compress(aid_list, [not p for p in are_part_aids])

    gids = ibs.get_annot_gids(list(set(part_aids)) + list(set(body_aids)))
    num_images = len(set(gids))
    assert num_images <= 1, "assign_parts_one_image called on multiple images' aids"

    # parallel lists representing all possible part/body pairs
    all_pairs_parallel = _all_pairs_parallel(part_aids, body_aids)
    pair_parts, pair_bodies = all_pairs_parallel

    if len(pair_parts) > 0 and len(pair_bodies) > 0:
        assigner_features = ibs.depc_annot.get(
            'assigner_viewpoint_features', all_pairs_parallel
        )
        # send all aids to this call just so it can find the right classifier model
        assigner_classifier = load_assigner_classifier(ibs, body_aids + part_aids)

        assigner_scores = assigner_classifier.predict_proba(assigner_features)
        #  assigner_scores is a list of [P_false, P_true] probabilities which sum to 1, so here we just pare down to the true probabilities
        assigner_scores = [score[1] for score in assigner_scores]
        good_pairs, unassigned_aids = _make_assignments(
            pair_parts, pair_bodies, assigner_scores, cutoff_score
        )
    else:
        good_pairs = []
        unassigned_aids = aid_list

    unassigned_aids = unassigned_aids_noconfig + unassigned_aids
    return good_pairs, unassigned_aids


def _make_assignments(pair_parts, pair_bodies, assigner_scores, cutoff_score=0.5):

    sorted_scored_pairs = [
        (part, body, score)
        for part, body, score in sorted(
            zip(pair_parts, pair_bodies, assigner_scores),
            key=lambda pbscore: pbscore[2],
            reverse=True,
        )
    ]

    assigned_pairs = []
    assigned_parts = set()
    assigned_bodies = set()
    n_bodies = len(set(pair_bodies))
    n_parts = len(set(pair_parts))
    n_true_pairs = min(n_bodies, n_parts)
    for part_aid, body_aid, score in sorted_scored_pairs:
        assign_this_pair = (
            part_aid not in assigned_parts
            and body_aid not in assigned_bodies
            and score >= cutoff_score
        )

        if assign_this_pair:
            assigned_pairs.append((part_aid, body_aid))
            assigned_parts.add(part_aid)
            assigned_bodies.add(body_aid)

        if (
            len(assigned_parts) is n_true_pairs
            or len(assigned_bodies) is n_true_pairs
            or score > cutoff_score
        ):
            break

    unassigned_parts = set(pair_parts) - set(assigned_parts)
    unassigned_bodies = set(pair_bodies) - set(assigned_bodies)
    unassigned_aids = sorted(list(unassigned_parts) + list(unassigned_bodies))

    return assigned_pairs, unassigned_aids


def load_assigner_classifier(ibs, aid_list, fallback_species='wild_dog'):
    species_with_part = ibs.get_annot_species(aid_list[0])
    species = species_with_part.split('+')[0]
    if species in INMEM_ASSIGNER_MODELS.keys():
        clf = INMEM_ASSIGNER_MODELS[species]
    else:
        if species not in SPECIES_CONFIG_MAP.keys():
            print(
                'WARNING: Assigner called for species %s which does not have an assigner modelfile specified. Falling back to the model for %s'
                % species,
                fallback_species,
            )
            species = fallback_species

        model_url = SPECIES_CONFIG_MAP[species]['model_url']
        model_fpath = ut.grab_file_url(model_url)
        from joblib import load

        clf = load(model_fpath)

    return clf


def illustrate_all_assignments(
    ibs,
    gid_to_assigner_results,
    gid_to_ground_truth,
    target_dir='/tmp/assigner-illustrations-2/',
    limit=20,
):

    correct_dir = os.path.join(target_dir, 'correct/')
    incorrect_dir = os.path.join(target_dir, 'incorrect/')

    for gid, assigned_aid_dict in gid_to_assigner_results.items()[:limit]:
        ground_t_dict = gid_to_ground_truth[gid]
        assigned_correctly = sorted(assigned_aid_dict['pairs']) == sorted(
            ground_t_dict['pairs']
        )
        if assigned_correctly:
            illustrate_assignments(
                ibs, gid, assigned_aid_dict, None, correct_dir
            )  # don't need to illustrate gtruth if it's identical to assignment
        else:
            illustrate_assignments(
                ibs, gid, assigned_aid_dict, ground_t_dict, incorrect_dir
            )

    print('illustrated assignments and saved them in %s' % target_dir)


# works on a single gid's worth of gid_keyed_assigner_results output
def illustrate_assignments(
    ibs,
    gid,
    assigned_aid_dict,
    gtruth_aid_dict,
    target_dir='/tmp/assigner-illustrations/',
):
    impath = ibs.get_image_paths(gid)
    imext = os.path.splitext(impath)[1]
    new_fname = os.path.join(target_dir, '%s%s' % (gid, imext))
    os.makedirs(target_dir, exist_ok=True)
    copy(impath, new_fname)

    with Image.open(new_fname) as image:
        _draw_all_annots(ibs, image, assigned_aid_dict, gtruth_aid_dict)
        image.save(new_fname)


def _draw_all_annots(ibs, image, assigned_aid_dict, gtruth_aid_dict):
    n_pairs = len(assigned_aid_dict['pairs'])
    # n_missing_pairs = 0
    #  TODO: missing pair shit
    n_unass = len(assigned_aid_dict['unassigned'])
    n_groups = n_pairs + n_unass
    colors = _pil_distinct_colors(n_groups)

    draw = ImageDraw.Draw(image)
    for i, pair in enumerate(assigned_aid_dict['pairs']):
        _draw_bbox(ibs, draw, pair[0], colors[i])
        _draw_bbox(ibs, draw, pair[1], colors[i])

    for i, aid in enumerate(assigned_aid_dict['unassigned'], start=n_pairs):
        _draw_bbox(ibs, draw, aid, colors[i])


def _pil_distinct_colors(n_colors):
    float_colors = pt.distinct_colors(n_colors)
    int_colors = [tuple([int(256 * f) for f in color]) for color in float_colors]
    return int_colors


def _draw_bbox(ibs, pil_draw, aid, color):
    verts = ibs.get_annot_rotated_verts(aid)
    pil_verts = [tuple(vertex) for vertex in verts]
    pil_verts += pil_verts[:1]  # for the line between the last and first vertex
    pil_draw.line(pil_verts, color, width=4)


def gid_keyed_assigner_results(ibs, all_pairs, all_unassigned_aids):
    one_from_each_pair = [p[0] for p in all_pairs]
    pair_gids = ibs.get_annot_gids(one_from_each_pair)
    unassigned_gids = ibs.get_annot_gids(all_unassigned_aids)

    gid_to_pairs = defaultdict(list)
    for pair, gid in zip(all_pairs, pair_gids):
        gid_to_pairs[gid] += [pair]

    gid_to_unassigned = defaultdict(list)
    for aid, gid in zip(all_unassigned_aids, unassigned_gids):
        gid_to_unassigned[gid] += [aid]

    gid_to_assigner_results = {}
    for gid in set(gid_to_pairs.keys()) | set(gid_to_unassigned.keys()):
        gid_to_assigner_results[gid] = {
            'pairs': gid_to_pairs[gid],
            'unassigned': gid_to_unassigned[gid],
        }

    return gid_to_assigner_results


def gid_keyed_ground_truth(ibs, assigner_data):
    test_pairs = assigner_data['test_pairs']
    test_truth = assigner_data['test_truth']
    assert len(test_pairs) == len(test_truth)

    aid_from_each_pair = [p[0] for p in test_pairs]
    gids_for_pairs = ibs.get_annot_gids(aid_from_each_pair)

    gid_to_pairs = defaultdict(list)
    gid_to_paired_aids = defaultdict(set)  # to know which have not been in any pair
    gid_to_all_aids = defaultdict(set)
    for pair, is_true_pair, gid in zip(test_pairs, test_truth, gids_for_pairs):
        gid_to_all_aids[gid] = gid_to_all_aids[gid] | set(pair)
        if is_true_pair:
            gid_to_pairs[gid] += [pair]
            gid_to_paired_aids[gid] = gid_to_paired_aids[gid] | set(pair)

    gid_to_unassigned_aids = defaultdict(list)
    for gid in gid_to_all_aids.keys():
        gid_to_unassigned_aids[gid] = list(gid_to_all_aids[gid] - gid_to_paired_aids[gid])

    gid_to_assigner_results = {}
    for gid in set(gid_to_pairs.keys()) | set(gid_to_unassigned_aids.keys()):
        gid_to_assigner_results[gid] = {
            'pairs': gid_to_pairs[gid],
            'unassigned': gid_to_unassigned_aids[gid],
        }

    return gid_to_assigner_results


@register_ibs_method
def assigner_testdb_ibs():
    import wbia
    from wbia import sysres

    dbdir = sysres.ensure_testdb_assigner()
    #  dbdir = '/data/testdb_assigner'
    ibs = wbia.opendb(dbdir=dbdir)
    return ibs


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.detect.assigner --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
