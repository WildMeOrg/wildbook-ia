# -*- coding: utf-8 -*-
"""
1) Ambiguity / num names
2) independence of annotations
3) continuous
4) exponential case
5) speicifc examples of our prob
6) human in loop
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
import numpy as np
from six.moves import zip
from wbia.unstable import pgm_ext

print, rrr, profile = ut.inject2(__name__)

# SPECIAL_BASIS_POOL = ['fred', 'sue', 'tom']
SPECIAL_BASIS_POOL = []
# 'fred', 'sue', 'tom']

# Quickly change names to be consistent with papers Sorry, person reading code.
# This will be confusing and inconsistent
NAME_TTYPE = 'name'
MATCH_TTYPE = 'same'
SCORE_TTYPE = 'evidence_match'


def temp_model(
    num_annots,
    num_names,
    score_evidence=[],
    name_evidence=[],
    other_evidence={},
    noquery=False,
    verbose=None,
    **kwargs
):
    if verbose is None:
        verbose = ut.VERBOSE

    method = kwargs.pop('method', None)
    model = make_name_model(num_annots, num_names, verbose=verbose, **kwargs)

    if verbose:
        model.print_priors(ignore_ttypes=[MATCH_TTYPE, SCORE_TTYPE])

    model, evidence, soft_evidence = update_model_evidence(
        model, name_evidence, score_evidence, other_evidence
    )

    if verbose and len(soft_evidence) != 0:
        model.print_priors(
            ignore_ttypes=[MATCH_TTYPE, SCORE_TTYPE], title='Soft Evidence', color='green'
        )

    # if verbose:
    #    ut.colorprint('\n --- Soft Evidence ---', 'white')
    #    for ttype, cpds in model.ttype2_cpds.items():
    #        if ttype != MATCH_TTYPE:
    #            for fs_ in ut.ichunks(cpds, 4):
    #                ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]),
    #                              'green')

    if verbose:
        ut.colorprint('\n --- Inference ---', 'red')

    if (len(evidence) > 0 or len(soft_evidence) > 0) and not noquery:
        evidence = model._ensure_internal_evidence(evidence)
        query_vars = []
        query_vars += ut.list_getattr(model.ttype2_cpds[NAME_TTYPE], 'variable')
        # query_vars += ut.list_getattr(model.ttype2_cpds[MATCH_TTYPE], 'variable')
        query_vars = ut.setdiff(query_vars, evidence.keys())
        # query_vars = ut.setdiff(query_vars, soft_evidence.keys())
        query_results = cluster_query(model, query_vars, evidence, soft_evidence, method)
    else:
        query_results = {}

    factor_list = query_results['factor_list']

    if verbose:
        if verbose:
            print('+--------')
        semtypes = [model.var2_cpd[f.variables[0]].ttype for f in factor_list]
        for type_, factors in ut.group_items(factor_list, semtypes).items():
            print('Result Factors (%r)' % (type_,))
            factors = ut.sortedby(factors, [f.variables[0] for f in factors])
            for fs_ in ut.ichunks(factors, 4):
                ut.colorprint(ut.hz_str([f._str('phi', 'psql') for f in fs_]), 'yellow')
        print('MAP assignments')
        top_assignments = query_results.get('top_assignments', [])
        tmp = []
        for lbl, val in top_assignments:
            tmp.append('%s : %.4f' % (ut.repr2(lbl), val))
        print(ut.align('\n'.join(tmp), ' :'))
        print('L_____\n')

    showkw = dict(evidence=evidence, soft_evidence=soft_evidence, **query_results)

    from wbia.algo.hots import pgm_viz

    pgm_viz.show_model(model, **showkw)
    return (model, evidence, query_results)
    # pgm_ext.print_ascii_graph(model)


def make_name_model(
    num_annots,
    num_names=None,
    verbose=True,
    mode=1,
    num_scores=2,
    p_score_given_same=None,
    hack_score_only=False,
    score_basis=None,
    special_names=None,
):
    r"""
    CommandLine:
        python -m wbia.algo.hots.bayes --exec-make_name_model --no-cnn
        python -m wbia.algo.hots.bayes --exec-make_name_model --show --no-cnn
        python -m wbia.algo.hots.bayes --exec-make_name_model --num-annots=3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.bayes import *  # NOQA
        >>> defaults = dict(num_annots=2, num_names=2, verbose=True)
        >>> modeltype = ut.get_argval('--modeltype', default='bayes')
        >>> kw = ut.argparse_funckw(make_name_model, defaults)
        >>> model = make_name_model(**kw)
        >>> ut.quit_if_noshow()
        >>> model.show_model(show_prior=False, show_title=False, modeltype=modeltype)
        >>> ut.show_if_requested()
    """
    if special_names is None:
        special_names = SPECIAL_BASIS_POOL

    assert mode == 1, 'only can do mode 1'
    base = ut.get_argval('--base', type_=str, default='a')
    annots = ut.chr_range(num_annots, base=base)
    # The indexes of match CPDs will not change if another annotation is added
    upper_diag_idxs = ut.colwise_diag_idxs(num_annots, 2)
    if hack_score_only:
        upper_diag_idxs = upper_diag_idxs[-hack_score_only:]

    if num_names is None:
        num_names = num_annots

    # +--- Define CPD Templates and Instantiation ---
    cpd_list = []

    # Name Factor
    name_cpd_t = pgm_ext.TemplateCPD(
        NAME_TTYPE, ('n', num_names), special_basis_pool=special_names
    )
    name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
    # name_cpds = [name_cpd_t.new_cpd(parents=aid, constrain_state=count)
    #             for count, aid in enumerate(annots, start=1)]
    cpd_list.extend(name_cpds)

    # Match Factor
    def match_pmf(match_type, n1, n2):
        return {True: {'same': 1.0, 'diff': 0.0}, False: {'same': 0.0, 'diff': 1.0}}[
            n1 == n2
        ][match_type]

    match_states = ['diff', 'same']
    match_cpd_t = pgm_ext.TemplateCPD(
        MATCH_TTYPE,
        match_states,
        evidence_ttypes=[name_cpd_t, name_cpd_t],
        pmf_func=match_pmf,
    )
    # match_cpd_t.varpref = 'S'
    namepair_cpds = ut.unflat_take(name_cpds, upper_diag_idxs)
    match_cpds = [match_cpd_t.new_cpd(parents=cpds) for cpds in namepair_cpds]
    cpd_list.extend(match_cpds)

    # Score Factor
    score_states = list(range(num_scores))
    if score_basis is not None:
        score_states = ['%.2f' % (s,) for s in score_basis]
    if p_score_given_same is None:
        tmp = np.arange(num_scores + 1)[1:]
        tmp = np.cumsum(tmp)
        tmp = tmp / tmp.sum()
        p_score_given_same = tmp

    def score_pmf(score_type, match_type):
        if isinstance(score_type, six.string_types):
            score_type = score_states.index(score_type)
        if match_type == 'same':
            return p_score_given_same[score_type]
        else:
            return p_score_given_same[-(score_type + 1)]

    score_cpd_t = pgm_ext.TemplateCPD(
        SCORE_TTYPE, score_states, evidence_ttypes=[match_cpd_t], pmf_func=score_pmf
    )
    # match_cpd_t.varpref = 'P'
    score_cpds = [score_cpd_t.new_cpd(parents=cpds) for cpds in zip(match_cpds)]
    cpd_list.extend(score_cpds)

    with_humans = False
    if with_humans:
        human_states = ['diff', 'same']
        human_cpd_t = pgm_ext.TemplateCPD(
            'human',
            human_states,
            evidence_ttypes=[match_cpd_t],
            pmf_func=[[0.9, 0.1], [0.1, 0.9]],
        )
        human_cpds = [human_cpd_t.new_cpd(parents=cpds) for cpds in zip(match_cpds)]
        cpd_list.extend(human_cpds)

    with_rank = False  # Rank depends on dependant scores
    if with_rank:
        rank_states = ['0', '1', '2', '3']
        rank_cpd_t = pgm_ext.TemplateCPD(
            'rank', rank_states, evidence_ttypes=[match_cpd_t], pmf_func=None
        )
        rank_cpds = [rank_cpd_t.new_cpd(parents=cpds) for cpds in zip(match_cpds)]
        cpd_list.extend(rank_cpds)

    # L___ End CPD Definitions ___

    print('score_cpds = %r' % (ut.list_getattr(score_cpds, 'variable'),))

    # Make Model
    model = pgm_ext.define_model(cpd_list)
    model.num_names = num_names

    if verbose:
        model.print_templates(ignore_ttypes=[MATCH_TTYPE])
    return model


def update_model_evidence(model, name_evidence, score_evidence, other_evidence):
    r"""

    CommandLine:
        python -m wbia.algo.hots.bayes --exec-update_model_evidence

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.bayes import *  # NOQA
        >>> verbose = True
        >>> other_evidence = {}
        >>> name_evidence = [0, 0, 1, 1, None]
        >>> score_evidence = ['high', 'low', 'low', 'low', 'low', 'high']
        >>> model = make_name_model(num_annots=5, num_names=3, verbose=True,
        >>>                         mode=1)
        >>> update_model_evidence(model, name_evidence, score_evidence,
        >>>                       other_evidence)
    """
    name_cpds = model.ttype2_cpds[NAME_TTYPE]
    score_cpds = model.ttype2_cpds[SCORE_TTYPE]

    evidence = {}
    evidence.update(other_evidence)
    soft_evidence = {}

    def apply_hard_soft_evidence(cpd_list, evidence_list):
        for cpd, ev in zip(cpd_list, evidence_list):
            if isinstance(ev, int):
                # hard internal evidence
                evidence[cpd.variable] = ev
            if isinstance(ev, six.string_types):
                # hard external evidence
                evidence[cpd.variable] = cpd._internal_varindex(cpd.variable, ev)
            if isinstance(ev, dict):
                # soft external evidence
                # HACK THAT MODIFIES CPD IN PLACE
                def rectify_evidence_val(_v, card=cpd.variable_card):
                    # rectify hacky string structures
                    tmp = 1 / (2 * card ** 2)
                    return (1 + tmp) / (card + tmp) if _v == '+eps' else _v

                ev_ = ut.map_dict_vals(rectify_evidence_val, ev)
                fill = (1.0 - sum(ev_.values())) / (cpd.variable_card - len(ev_))
                # HACK fix for float problems
                if len(ev_) == cpd.variable_card - 1:
                    fill = 0

                assert fill > -1e7, 'fill=%r' % (fill,)
                row_labels = list(ut.iprod(*cpd.statenames))

                for i, lbl in enumerate(row_labels):
                    if lbl in ev_:
                        # external case1
                        cpd.values[i] = ev_[lbl]
                    elif len(lbl) == 1 and lbl[0] in ev_:
                        # external case2
                        cpd.values[i] = ev_[lbl[0]]
                    elif i in ev_:
                        # internal case
                        cpd.values[i] = ev_[i]
                    else:
                        cpd.values[i] = fill
                cpd.normalize()
                soft_evidence[cpd.variable] = True

    apply_hard_soft_evidence(name_cpds, name_evidence)
    apply_hard_soft_evidence(score_cpds, score_evidence)
    return model, evidence, soft_evidence


def reduce_marginalize(phi, query_variables=None, evidence={}, inplace=False):
    """
    Hack for reduction followed by marginalization

    Example:
        >>> reduced_joint = joint.observe(
        >>>     query_variables, evidence, inplace=False)
        >>> new_rows = reduced_joint._row_labels()
        >>> new_vals = reduced_joint.values.ravel()
        >>> map_vals = new_rows[new_vals.argmax()]
        >>> map_assign = dict(zip(reduced_joint.variables, map_vals))
    """
    reduced_joint = phi if inplace else phi.copy()
    if query_variables is None:
        query_variables = reduced_joint.variables
    reduced_joint.reduce(evidence)
    reduced_joint.normalize()
    # Marginalize over non-query, non-evidence
    irrelevant_vars = set(reduced_joint.variables) - (
        set(evidence.keys()) | set(query_variables)
    )
    reduced_joint.marginalize(irrelevant_vars)
    reduced_joint.normalize()
    if not inplace:
        return reduced_joint


def make_temp_state(state):
    mapping = {}
    for state_idx in state:
        if state_idx not in mapping:
            mapping[state_idx] = -(len(mapping) + 1)
    temp_state = [mapping[state_idx] for state_idx in state]
    return temp_state


def collapse_labels(model, evidence, reduced_variables, reduced_row_idxs, reduced_values):
    import vtool as vt

    # assert np.all(reduced_joint.values.ravel() == reduced_joint.values.flatten())
    reduced_ttypes = [model.var2_cpd[var].ttype for var in reduced_variables]

    evidence_vars = list(evidence.keys())
    evidence_state_idxs = ut.dict_take(evidence, evidence_vars)
    evidence_ttypes = [model.var2_cpd[var].ttype for var in evidence_vars]

    ttype2_ev_indices = dict(zip(*ut.group_indices(evidence_ttypes)))
    ttype2_re_indices = dict(zip(*ut.group_indices(reduced_ttypes)))
    # ttype2_ev_indices = ut.group_items(range(len(evidence_vars)), evidence_ttypes)
    # ttype2_re_indices = ut.group_items(range(len(reduced_variables)), reduced_ttypes)

    # Allow specific types of labels to change
    # everything is the same, only the names have changed.
    # TODO: allow for multiple different label_ttypes
    # for label_ttype in label_ttypes
    if NAME_TTYPE not in model.ttype2_template:
        return reduced_row_idxs, reduced_values
    label_ttypes = [NAME_TTYPE]
    for label_ttype in label_ttypes:
        ev_colxs = ttype2_ev_indices[label_ttype]
        re_colxs = ttype2_re_indices[label_ttype]

        ev_state_idxs = ut.take(evidence_state_idxs, ev_colxs)
        ev_state_idxs_tile = np.tile(ev_state_idxs, (len(reduced_values), 1)).astype(
            np.int
        )
        num_ev_ = len(ev_colxs)

        aug_colxs = list(range(num_ev_)) + (np.array(re_colxs) + num_ev_).tolist()
        aug_state_idxs = np.hstack([ev_state_idxs_tile, reduced_row_idxs])

        # Relabel rows based on the knowledge that
        # everything is the same, only the names have changed.

        num_cols = len(aug_state_idxs.T)
        mask = vt.index_to_boolmask(aug_colxs, num_cols)
        (other_colxs,) = np.where(~mask)
        relbl_states = aug_state_idxs.compress(mask, axis=1)
        other_states = aug_state_idxs.compress(~mask, axis=1)
        tmp_relbl_states = np.array(list(map(make_temp_state, relbl_states)))

        max_tmp_state = -1
        min_tmp_state = tmp_relbl_states.min()

        # rebuild original state structure with temp state idxs
        tmp_state_cols = [None] * num_cols
        for count, colx in enumerate(aug_colxs):
            tmp_state_cols[colx] = tmp_relbl_states[:, count : count + 1]
        for count, colx in enumerate(other_colxs):
            tmp_state_cols[colx] = other_states[:, count : count + 1]
        tmp_state_idxs = np.hstack(tmp_state_cols)

        data_ids = np.array(vt.compute_unique_data_ids_(list(map(tuple, tmp_state_idxs))))
        unique_ids, groupxs = vt.group_indices(data_ids)
        print('Collapsed %r states into %r states' % (len(data_ids), len(unique_ids),))
        # Sum the values in the cpd to marginalize the duplicate probs
        new_values = np.array(
            [g.sum() for g in vt.apply_grouping(reduced_values, groupxs)]
        )
        # Take only the unique rows under this induced labeling
        unique_tmp_groupxs = np.array(ut.get_list_column(groupxs, 0))
        new_aug_state_idxs = tmp_state_idxs.take(unique_tmp_groupxs, axis=0)

        tmp_idx_set = set((-np.arange(-max_tmp_state, (-min_tmp_state) + 1)).tolist())
        true_idx_set = set(range(len(model.ttype2_template[label_ttype].basis)))

        # Relabel the rows one more time to agree with initial constraints
        for colx, true_idx in enumerate(ev_state_idxs):
            tmp_idx = np.unique(new_aug_state_idxs.T[colx])
            assert len(tmp_idx) == 1
            tmp_idx_set -= {tmp_idx[0]}
            true_idx_set -= {true_idx}
            new_aug_state_idxs[new_aug_state_idxs == tmp_idx] = true_idx
        # Relabel the remaining idxs
        remain_tmp_idxs = sorted(list(tmp_idx_set))[::-1]
        remain_true_idxs = sorted(list(true_idx_set))
        for tmp_idx, true_idx in zip(remain_tmp_idxs, remain_true_idxs):
            new_aug_state_idxs[new_aug_state_idxs == tmp_idx] = true_idx

        # Remove evidence based augmented labels
        new_state_idxs = new_aug_state_idxs.T[num_ev_:].T
        return new_state_idxs, new_values


def collapse_factor_labels(model, reduced_joint, evidence):
    reduced_variables = reduced_joint.variables
    reduced_row_idxs = np.array(reduced_joint._row_labels(asindex=True))
    reduced_values = reduced_joint.values.ravel()

    new_state_idxs, new_values = collapse_labels(
        model, evidence, reduced_variables, reduced_row_idxs, reduced_values
    )

    if isinstance(reduced_joint, pgm_ext.ApproximateFactor):
        new_reduced_joint = pgm_ext.ApproximateFactor(
            new_state_idxs,
            new_values,
            reduced_variables,
            statename_dict=reduced_joint.statename_dict,
        )
    else:
        # hack into a new joint factor
        # (that is the same size as the reduced_joint)
        new_reduced_joint = reduced_joint.copy()
        assert new_reduced_joint.values is not reduced_joint.values, 'copy did not work'
        new_reduced_joint.values[:] = 0
        flat_idxs = np.ravel_multi_index(new_state_idxs.T, new_reduced_joint.values.shape)

        old_values = new_reduced_joint.values.ravel()
        old_values[flat_idxs] = new_values
        new_reduced_joint.values = old_values.reshape(reduced_joint.cardinality)
        # print(new_reduced_joint._str(maxrows=4, sort=-1))
    # return new_reduced_joint, new_state_idxs, new_values
    return new_reduced_joint


def report_partitioning_statistics(new_reduced_joint):
    # compute partitioning statistics
    import vtool as vt

    vals, idxs = vt.group_indices(new_reduced_joint.values.ravel())
    # groupsize = list(map(len, idxs))
    # groupassigns = ut.unflat_vecmap(new_reduced_joint.assignment, idxs)
    all_states = new_reduced_joint._row_labels(asindex=True)
    clusterstats = [tuple(sorted(list(ut.dict_hist(a).values()))) for a in all_states]
    grouped_vals = ut.group_items(new_reduced_joint.values.ravel(), clusterstats)

    # probs_assigned_to_clustertype = [(
    #    sorted(np.unique(np.array(b).round(decimals=5)).tolist())[::-1], a)
    #    for a, b in grouped_vals.items()]
    probs_assigned_to_clustertype = [
        (ut.dict_hist(np.array(b).round(decimals=5)), a) for a, b in grouped_vals.items()
    ]
    sortx = ut.argsort([max(c[0].keys()) for c in probs_assigned_to_clustertype])
    probs_assigned_to_clustertype = ut.take(probs_assigned_to_clustertype, sortx)

    # This list of 2-tuples with the first item being the unique
    # probabilies that are assigned to a cluster type along with the number
    # of times they were assigned. A cluster type is the second item. Every
    # number represents how many annotations were assigned to a specific
    # label. The length of that list is the number of total labels.  For
    # all low scores you will see [[{somenum: 1}, {0: 800}], [1, 1, 1, ... 1]]
    # indicating that that the assignment of everyone to a different label happend once
    # where the probability was somenum and a 800 times where the probability was 0.

    # print(sorted([(b, a) for a, b in ut.map_dict_vals(sum, x)]).items())
    # z = sorted([(b, a) for a, b in ut.map_dict_vals(sum, grouped_vals).items()])
    print(ut.repr2(probs_assigned_to_clustertype, nl=2, precision=2, sorted_=True))

    # group_numperlbl = [
    #    [sorted(list(ut.dict_hist(ut.get_list_column(a, 1)).values())) for a in assigns]
    #    for assigns in groupassigns]


def _test_compute_reduced_joint(model, query_vars, evidence, method):
    import pgmpy

    operation = 'maximize'
    variables = query_vars

    infr_ve = pgmpy.inference.VariableElimination(model)
    joint_ve = infr_ve.compute_joint(variables, operation, evidence)
    joint_ve.normalize()
    joint_ve.reorder()

    infr_bp = pgmpy.inference.BeliefPropagation(model)
    joint_bp = infr_bp.compute_joint(variables, operation, evidence)
    joint_bp.normalize()
    joint_bp.reorder()

    assert np.allclose(joint_ve.values, joint_bp.values)
    print('VE and BP are the same')

    joint_bf = model.joint_distribution()
    reduce_marginalize(joint_bf, query_vars, evidence, inplace=True)

    assert np.allclose(joint_bf.values, joint_bp.values)
    print('BF and BP are the same')


def compute_reduced_joint(model, query_vars, evidence, method, operation='maximize'):
    import pgmpy

    if method == 'approx':
        # TODO: incorporate operation?
        query_states = model.get_number_of_states(query_vars)
        print('model.number_of_states = %r' % (model.get_number_of_states(),))
        print('query_states = %r' % (query_states,))
        # Try to approximatly sample the map inference
        infr = pgmpy.inference.Sampling.BayesianModelSampling(model)

        # The markov blanket of a name node in our network
        # can be quite large. It includes all other names.

        # infr = pgmpy.inference.Sampling.GibbsSampling(model)
        # import utool
        # utool.embed()

        # infr = pgmpy.inference.Sampling.GibbsSampling()
        # infr._get_kernel_from_bayesian_model(model)

        evidence_ = [pgmpy.inference.Sampling.State(*item) for item in evidence.items()]

        # TODO: apply hoffding and chernoff bounds
        delta = 0.1  # desired probability of error
        eps = 0.2  # desired error bound

        u = 1 / (2 ** len(evidence))  # upper bound on cpd entries of evidence
        k = len(evidence)
        gamma = (4 * (1 + eps) / (eps ** 2)) * np.log(2 / delta)

        thresh = gamma * (u ** k)

        # We are observing the leaves of this network, which means
        # we are effectively sampling from the prior distribution
        # when using forward sampling.

        Py = 1 / query_states
        Py_hueristic = 1 / (4 ** len(query_vars))
        M_hoffding = np.log(2 / delta) / (2 * eps ** 2)
        M_chernoff = 3 * (np.log(2 / delta) / (Py * eps ** 2))
        M_chernoff_hueristic = 3 * (np.log(2 / delta) / (Py_hueristic * eps ** 2))
        hueristic_size = 2 ** (len(query_vars) + 2)
        size = min(100000, max(hueristic_size, 128))
        print('\n-----')
        print('u = %r' % (u,))
        print('thresh = %r' % (thresh,))
        print('k = %r' % (k,))
        print('gamma = %r' % (gamma,))
        print('M_chernoff_hueristic = %r' % (M_chernoff_hueristic,))
        print('hueristic_size = %r' % (hueristic_size,))
        print('M_hoffding = %r' % (M_hoffding,))
        print('M_chernoff = %r' % (M_chernoff,))
        print('size = %r' % (size,))
        # np.log(2 / .1) / (2 * (.2 ** 2))
        sampled = infr.likelihood_weighted_sample(evidence=evidence_, size=size)
        reduced_joint = pgm_ext.ApproximateFactor.from_sampled(
            sampled, query_vars, statename_dict=model.statename_dict
        )
        # self = reduced_joint  # NOQA
        # arr = self.state_idxs  # NOQA
        # import utool
        # utool.embed()
        num_raw_states = len(reduced_joint.state_idxs)
        reduced_joint.consolidate()
        num_unique_states = len(reduced_joint.state_idxs)
        print(
            '[pgm] %r / %r initially sampled states are unique'
            % (num_unique_states, num_raw_states,)
        )
        reduced_joint.normalize()
        reduced_joint.reorder()
    elif method == 'varelim':
        infr = pgmpy.inference.VariableElimination(model)
        reduced_joint = infr.compute_joint(query_vars, operation, evidence)
        reduced_joint.normalize()
        reduced_joint.reorder()
    elif method in ['bp', 'beliefprop']:
        # Dont brute force anymore
        infr = pgmpy.inference.BeliefPropagation(model)
        reduced_joint = infr.compute_joint(query_vars, operation, evidence)
        reduced_joint.normalize()
        reduced_joint.reorder()
    elif method in ['bf', 'brute', 'bruteforce']:
        # TODO: incorporate operation?

        full_joint = model.joint_distribution()
        reduced_joint = reduce_marginalize(
            full_joint, query_vars, evidence, inplace=False
        )
        del full_joint
    else:
        raise NotImplementedError('method=%r' % (method,))
    return reduced_joint


def cluster_query(
    model,
    query_vars=None,
    evidence=None,
    soft_evidence=None,
    method=None,
    operation='maximize',
):
    """
    CommandLine:
        python -m wbia.algo.hots.bayes --exec-cluster_query --show

    GridParams:
        >>> param_grid = dict(
        >>>     #method=['approx', 'bf', 'bp'],
        >>>     method=['approx', 'bp'],
        >>> )
        >>> combos = ut.all_dict_combinations(param_grid)
        >>> index = 0
        >>> keys = 'method'.split(', ')
        >>> method, = ut.dict_take(combos[index], keys)

    GridSetup:
        >>> from wbia.algo.hots.bayes import *  # NOQA
        >>> verbose = True
        >>> other_evidence = {}
        >>> name_evidence = [1, None, None, 0]
        >>> score_evidence = [2, 0, 2]
        >>> special_names = ['fred', 'sue', 'tom', 'paul']
        >>> model = make_name_model(
        >>>     num_annots=4, num_names=4, num_scores=3, verbose=True, mode=1,
        >>>     special_names=special_names)
        >>> method = None
        >>> model, evidence, soft_evidence = update_model_evidence(
        >>>     model, name_evidence, score_evidence, other_evidence)
        >>> evidence = model._ensure_internal_evidence(evidence)
        >>> query_vars = ut.list_getattr(model.ttype2_cpds[NAME_TTYPE], 'variable')

    GridExample:
        >>> # DISABLE_DOCTEST
        >>> query_results = cluster_query(model, query_vars, evidence,
        >>>                               method=method)
        >>> print(ut.repr2(query_results['top_assignments'], nl=1))
        >>> ut.quit_if_noshow()
        >>> from wbia.algo.hots import pgm_viz
        >>> pgm_viz.show_model(model, evidence=evidence, **query_results)
        >>> ut.show_if_requested()
    """
    evidence = model._ensure_internal_evidence(evidence)
    if query_vars is None:
        query_vars = model.nodes()
    orig_query_vars = query_vars  # NOQA
    query_vars = ut.setdiff(query_vars, list(evidence.keys()))

    if method is None:
        method = ut.get_argval('--method', type_=str, default='bp')

    reduced_joint = compute_reduced_joint(model, query_vars, evidence, method, operation)

    new_reduced_joint = collapse_factor_labels(model, reduced_joint, evidence)

    if False:
        report_partitioning_statistics(new_reduced_joint)

    # FIXME: are these max marginals?
    max_marginals = {}
    for i, var in enumerate(query_vars):
        one_out = query_vars[:i] + query_vars[i + 1 :]
        max_marginals[var] = new_reduced_joint.marginalize(one_out, inplace=False)
        # max_marginals[var] = joint2.maximize(one_out, inplace=False)
    factor_list = max_marginals.values()

    # Now find the most likely state
    reduced_variables = new_reduced_joint.variables
    new_state_idxs = np.array(new_reduced_joint._row_labels(asindex=True))
    new_values = new_reduced_joint.values.ravel()
    sortx = new_values.argsort()[::-1]
    sort_new_state_idxs = new_state_idxs.take(sortx, axis=0)
    sort_new_values = new_values.take(sortx)
    sort_new_states = list(
        zip(
            *[
                ut.dict_take(model.statename_dict[var], idx)
                for var, idx in zip(reduced_variables, sort_new_state_idxs.T)
            ]
        )
    )

    # Better map assignment based on knowledge of labels
    map_assign = dict(zip(reduced_variables, sort_new_states[0]))

    sort_reduced_rowstr_lbls = [
        ut.repr2(
            dict(zip(reduced_variables, lbls)), explicit=True, nobraces=True, strvals=True
        )
        for lbls in sort_new_states
    ]

    top_assignments = list(zip(sort_reduced_rowstr_lbls[:4], sort_new_values))
    if len(sort_new_values) > 3:
        top_assignments += [('other', 1 - sum(sort_new_values[:4]))]
    query_results = {
        'factor_list': factor_list,
        'top_assignments': top_assignments,
        'map_assign': map_assign,
        'method': method,
    }
    print('query_results = %s' % (ut.repr3(query_results, nl=2),))
    return query_results


def draw_tree_model(model, **kwargs):
    import wbia.plottool as pt
    import networkx as netx

    if not ut.get_argval('--hackjunc'):
        fnum = pt.ensure_fnum(None)
        fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
        ax = pt.gca()
        # name_nodes = sorted(ut.list_getattr(model.ttype2_cpds[NAME_TTYPE], 'variable'))
        netx_graph = model.to_markov_model()
        # pos = netx.pygraphviz_layout(netx_graph)
        # pos = netx.graphviz_layout(netx_graph)
        # pos = get_hacked_pos(netx_graph, name_nodes, prog='neato')
        pos = netx.nx_pydot.pydot_layout(netx_graph)
        node_color = [pt.WHITE] * len(pos)
        drawkw = dict(
            pos=pos, ax=ax, with_labels=True, node_color=node_color, node_size=1100
        )
        netx.draw(netx_graph, **drawkw)
        if kwargs.get('show_title', True):
            pt.set_figtitle('Markov Model')

    if not ut.get_argval('--hackmarkov'):
        fnum = pt.ensure_fnum(None)
        fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
        ax = pt.gca()
        netx_graph = model.to_junction_tree()

        # prettify nodes

        def fixtupkeys(dict_):
            return {
                ', '.join(k) if isinstance(k, tuple) else k: fixtupkeys(v)
                for k, v in dict_.items()
            }

        # FIXME
        n = fixtupkeys(netx_graph.node)
        e = fixtupkeys(netx_graph.edge)
        a = fixtupkeys(netx_graph.adj)
        netx_graph.nodes.update(n)
        netx_graph.edges.update(e)
        netx_graph.adj.update(a)
        # netx_graph = model.to_markov_model()
        # pos = netx.pygraphviz_layout(netx_graph)
        # pos = netx.graphviz_layout(netx_graph)
        pos = netx.nx_pydot.pydot_layout(netx_graph)
        node_color = [pt.WHITE] * len(pos)
        drawkw = dict(
            pos=pos, ax=ax, with_labels=True, node_color=node_color, node_size=2000
        )
        netx.draw(netx_graph, **drawkw)
        if kwargs.get('show_title', True):
            pt.set_figtitle('Junction/Clique Tree / Cluster Graph')


def get_hacked_pos(netx_graph, name_nodes=None, prog='dot'):
    import pygraphviz
    import networkx as netx

    # Add "invisible" edges to induce an ordering
    # Hack for layout (ordering of top level nodes)
    netx_graph2 = netx_graph.copy()
    if getattr(netx_graph, 'ttype2_cpds', None) is not None:
        grouped_nodes = []
        for ttype in netx_graph.ttype2_cpds.keys():
            ttype_cpds = netx_graph.ttype2_cpds[ttype]
            # use defined ordering
            ttype_nodes = ut.list_getattr(ttype_cpds, 'variable')
            # ttype_nodes = sorted(ttype_nodes)
            invis_edges = list(ut.itertwo(ttype_nodes))
            netx_graph2.add_edges_from(invis_edges)
            grouped_nodes.append(ttype_nodes)

        A = netx.to_agraph(netx_graph2)
        for nodes in grouped_nodes:
            A.add_subgraph(nodes, rank='same')
    else:
        A = netx.to_agraph(netx_graph2)

    # if name_nodes is not None:
    #    #netx.set_node_attributes(netx_graph, name='label', values={n: {'label': n} for n in all_nodes})
    #    invis_edges = list(ut.itertwo(name_nodes))
    #    netx_graph2.add_edges_from(invis_edges)
    #    A.add_subgraph(name_nodes, rank='same')
    # else:
    #    A = netx.to_agraph(netx_graph2)
    args = ''
    G = netx_graph
    A.layout(prog=prog, args=args)
    # A.draw('example.png', prog='dot')
    node_pos = {}
    for n in G:
        node_ = pygraphviz.Node(A, n)
        try:
            xx, yy = node_.attr['pos'].split(',')
            node_pos[n] = (float(xx), float(yy))
        except Exception:
            print('no position for node', n)
            node_pos[n] = (0.0, 0.0)
    return node_pos


def show_model(model, evidence={}, soft_evidence={}, **kwargs):
    """
    References:
        http://stackoverflow.com/questions/22207802/pygraphviz-networkx-set-node-level-or-layer

    Ignore:
        pkg-config --libs-only-L libcgraph
        sudo apt-get  install libgraphviz-dev -y
        sudo apt-get  install libgraphviz4 -y

        # sudo apt-get install pkg-config
        sudo apt-get install libgraphviz-dev
        # pip install git+git://github.com/pygraphviz/pygraphviz.git
        pip install pygraphviz
        python -c "import pygraphviz; print(pygraphviz.__file__)"

        sudo pip3 install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
        python3 -c "import pygraphviz; print(pygraphviz.__file__)"

    CommandLine:
        python -m wbia.algo.hots.bayes --exec-show_model --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.bayes import *  # NOQA
        >>> model = '?'
        >>> evidence = {}
        >>> soft_evidence = {}
        >>> result = show_model(model, evidence, soft_evidence)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    if ut.get_argval('--hackmarkov') or ut.get_argval('--hackjunc'):
        draw_tree_model(model, **kwargs)
        return

    import wbia.plottool as pt
    import networkx as netx

    fnum = pt.ensure_fnum(None)
    netx_graph = model
    # netx_graph.graph.setdefault('graph', {})['size'] = '"10,5"'
    # netx_graph.graph.setdefault('graph', {})['rankdir'] = 'LR'

    pos_dict = get_hacked_pos(netx_graph)
    # pos_dict = netx.nx_agraph.pygraphviz_layout(netx_graph)
    # pos = netx.nx_agraph.nx_pydot.pydot_layout(netx_graph, prog='dot')
    # pos_dict = netx.nx_agraph.graphviz_layout(netx_graph)

    textprops = {
        'family': 'monospace',
        'horizontalalignment': 'left',
        # 'horizontalalignment': 'center',
        # 'size': 12,
        'size': 8,
    }

    netx_nodes = model.nodes(data=True)
    node_key_list = ut.get_list_column(netx_nodes, 0)
    pos_list = ut.dict_take(pos_dict, node_key_list)

    var2_post = {f.variables[0]: f for f in kwargs.get('factor_list', [])}

    prior_text = None
    post_text = None
    evidence_tas = []
    post_tas = []
    prior_tas = []
    node_color = []

    has_inferred = evidence or var2_post
    if has_inferred:
        ignore_prior_with_ttype = [SCORE_TTYPE, MATCH_TTYPE]
        show_prior = False
    else:
        ignore_prior_with_ttype = []
        # show_prior = True
        show_prior = False

    dpy = 5
    dbx, dby = (20, 20)
    takw1 = {'bbox_align': (0.5, 0), 'pos_offset': [0, dpy], 'bbox_offset': [dbx, dby]}
    takw2 = {'bbox_align': (0.5, 1), 'pos_offset': [0, -dpy], 'bbox_offset': [-dbx, -dby]}

    name_colors = pt.distinct_colors(max(model.num_names, 10))
    name_colors = name_colors[: model.num_names]

    # cmap_ = 'hot' #mx = 0.65 #mn = 0.15
    cmap_, mn, mx = 'plasma', 0.15, 1.0
    _cmap = pt.plt.get_cmap(cmap_)

    def cmap(x):
        return _cmap((x * mx) + mn)

    for node, pos in zip(netx_nodes, pos_list):
        variable = node[0]
        cpd = model.var2_cpd[variable]
        prior_marg = (
            cpd if cpd.evidence is None else cpd.marginalize(cpd.evidence, inplace=False)
        )

        show_evidence = variable in evidence
        show_prior = cpd.ttype not in ignore_prior_with_ttype
        show_post = variable in var2_post
        show_prior |= cpd.ttype not in ignore_prior_with_ttype

        post_marg = None

        if show_post:
            post_marg = var2_post[variable]

        def get_name_color(phi):
            order = phi.values.argsort()[::-1]
            if len(order) < 2:
                dist_next = phi.values[order[0]]
            else:
                dist_next = phi.values[order[0]] - phi.values[order[1]]
            dist_total = phi.values[order[0]]
            confidence = (dist_total * dist_next) ** (2.5 / 4)
            # print('confidence = %r' % (confidence,))
            color = name_colors[order[0]]
            color = pt.color_funcs.desaturate_rgb(color, 1 - confidence)
            color = np.array(color)
            return color

        if variable in evidence:
            if cpd.ttype == SCORE_TTYPE:
                cmap_index = evidence[variable] / (cpd.variable_card - 1)
                color = cmap(cmap_index)
                color = pt.lighten_rgb(color, 0.4)
                color = np.array(color)
                node_color.append(color)
            elif cpd.ttype == NAME_TTYPE:
                color = name_colors[evidence[variable]]
                color = np.array(color)
                node_color.append(color)
            else:
                color = pt.FALSE_RED
                node_color.append(color)
        # elif variable in soft_evidence:
        #    color = pt.LIGHT_PINK
        #    show_prior = True
        #    color = get_name_color(prior_marg)
        #    node_color.append(color)
        else:
            if cpd.ttype == NAME_TTYPE and post_marg is not None:
                color = get_name_color(post_marg)
                node_color.append(color)
            elif cpd.ttype == MATCH_TTYPE and post_marg is not None:
                color = cmap(post_marg.values[1])
                color = pt.lighten_rgb(color, 0.4)
                color = np.array(color)
                node_color.append(color)
            else:
                # color = pt.WHITE
                color = pt.NEUTRAL
                node_color.append(color)

        if show_prior:
            if variable in soft_evidence:
                prior_color = pt.LIGHT_PINK
            else:
                prior_color = None
            prior_text = pgm_ext.make_factor_text(prior_marg, 'prior')
            prior_tas.append(dict(text=prior_text, pos=pos, color=prior_color, **takw2))
        if show_evidence:
            _takw1 = takw1
            if cpd.ttype == SCORE_TTYPE:
                _takw1 = takw2
            evidence_text = cpd.variable_statenames[evidence[variable]]
            if isinstance(evidence_text, int):
                evidence_text = '%d/%d' % (evidence_text + 1, cpd.variable_card)
            evidence_tas.append(dict(text=evidence_text, pos=pos, color=color, **_takw1))
        if show_post:
            _takw1 = takw1
            if cpd.ttype == MATCH_TTYPE:
                _takw1 = takw2
            post_text = pgm_ext.make_factor_text(post_marg, 'post')
            post_tas.append(dict(text=post_text, pos=pos, color=None, **_takw1))

    def trnps_(dict_list):
        """ tranpose dict list """
        list_dict = ut.ddict(list)
        for dict_ in dict_list:
            for key, val in dict_.items():
                list_dict[key + '_list'].append(val)
        return list_dict

    takw1_ = trnps_(post_tas + evidence_tas)
    takw2_ = trnps_(prior_tas)

    # Draw graph
    if has_inferred:
        pnum1 = (3, 1, (slice(0, 2), 0))
    else:
        pnum1 = None

    fig = pt.figure(fnum=fnum, pnum=pnum1, doclf=True)  # NOQA
    ax = pt.gca()
    # print('node_color = %s' % (ut.repr3(node_color),))
    drawkw = dict(
        pos=pos_dict, ax=ax, with_labels=True, node_size=1500, node_color=node_color
    )
    netx.draw(netx_graph, **drawkw)

    hacks = []
    if len(post_tas + evidence_tas):
        hacks.append(pt.draw_text_annotations(textprops=textprops, **takw1_))
    if prior_tas:
        hacks.append(pt.draw_text_annotations(textprops=textprops, **takw2_))

    xmin, ymin = np.array(pos_list).min(axis=0)
    xmax, ymax = np.array(pos_list).max(axis=0)
    num_annots = len(model.ttype2_cpds[NAME_TTYPE])
    if num_annots > 4:
        ax.set_xlim((xmin - 40, xmax + 40))
        ax.set_ylim((ymin - 50, ymax + 50))
        fig.set_size_inches(30, 7)
    else:
        ax.set_xlim((xmin - 42, xmax + 42))
        ax.set_ylim((ymin - 50, ymax + 50))
        fig.set_size_inches(23, 7)
    fig = pt.gcf()

    title = 'num_names=%r, num_annots=%r' % (model.num_names, num_annots,)
    map_assign = kwargs.get('map_assign', None)

    top_assignments = kwargs.get('top_assignments', None)
    if top_assignments is not None:
        map_assign, map_prob = top_assignments[0]
        if map_assign is not None:

            def word_insert(text):
                return '' if len(text) == 0 else text + ' '

            title += '\n%sMAP: ' % (word_insert(kwargs.get('method', '')))
            title += map_assign + ' @' + '%.2f%%' % (100 * map_prob,)
    if kwargs.get('show_title', True):
        pt.set_figtitle(title, size=14)

    for hack in hacks:
        hack()

    # Hack in colorbars
    if has_inferred:
        pt.colorbar(
            np.linspace(0, 1, len(name_colors)),
            name_colors,
            lbl=NAME_TTYPE,
            ticklabels=model.ttype2_template[NAME_TTYPE].basis,
            ticklocation='left',
        )

        basis = model.ttype2_template[SCORE_TTYPE].basis
        scalars = np.linspace(0, 1, len(basis))
        scalars = np.linspace(0, 1, 100)
        colors = pt.scores_to_color(
            scalars, cmap_=cmap_, reverse_cmap=False, cmap_range=(mn, mx)
        )
        colors = [pt.lighten_rgb(c, 0.4) for c in colors]

        if ut.list_type(basis) is int:
            pt.colorbar(scalars, colors, lbl=SCORE_TTYPE, ticklabels=np.array(basis) + 1)
        else:
            pt.colorbar(scalars, colors, lbl=SCORE_TTYPE, ticklabels=basis)
            # print('basis = %r' % (basis,))

    # Draw probability hist
    if has_inferred and top_assignments is not None:
        bin_labels = ut.get_list_column(top_assignments, 0)
        bin_vals = ut.get_list_column(top_assignments, 1)

        # bin_labels = ['\n'.join(ut.textwrap.wrap(_lbl, width=30)) for _lbl in bin_labels]

        pt.draw_histogram(
            bin_labels,
            bin_vals,
            fnum=fnum,
            pnum=(3, 8, (2, slice(4, None))),
            transpose=True,
            use_darkbackground=False,
            # xtick_rotation=-10,
            ylabel='Prob',
            xlabel='assignment',
        )
        pt.set_title('Assignment probabilities')
    # fpath = ('name_model_' + suff + '.png')
    # pt.plt.savefig(fpath)
    # return fpath


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.hots.bayes
        python -m wbia.algo.hots.bayes --allexamples
    """
    if ut.VERBOSE:
        print('[hs] bayes')
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
